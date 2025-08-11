# %%
import os
import sys
sys.path.append('/app/Craftax/craftax')
os.environ["CUDA_VISIBLE_DEVICES"] = "5,"

import jax
from craftax_marl.envs.craftax_symbolic_env import CraftaxMARLSymbolicEnv as CraftaxEnv
from craftax_marl.renderer.renderer_text import render_craftax_text
from craftax_marl.renderer.text_helper import action_map

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import re
from collections import defaultdict

rng = jax.random.PRNGKey(5)
env = CraftaxEnv()

# %%
# ── Base prompt: constant instruction + per-agent memory + changing observation ─
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{instruction}"),
        MessagesPlaceholder("history"),      # <- where each agent's memory goes
        ("human", "{observation}")           # <- changing observation each call
    ]
)

# %%
API_KEY = ""
MODEL_NAME = "gpt-4o"
llm = ChatOpenAI(
    model=MODEL_NAME, 
    api_key=API_KEY
)

chain = prompt | llm

# %%
from langchain_core.chat_history import InMemoryChatMessageHistory
# ── Memory store: one ChatMessageHistory per agent ─────────────────────────────
# We'll map session_id -> ChatMessageHistory. Each agent uses a unique session_id.
histories = defaultdict(InMemoryChatMessageHistory)

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    return histories[session_id]

agent_chain = RunnableWithMessageHistory(
    chain,
    get_session_history=get_history,
    input_messages_key="observation",
    history_messages_key="history",
)

def ask_agent(session_id: str, observation: str, instruction: str):
    """
    Sends a changing observation to a specific agent.
    Each agent maintains its own memory via its session_id.
    """
    config = {"configurable": {"session_id": session_id}}
    return agent_chain.invoke(
        {"instruction": instruction, "observation": observation},
        config=config
    )
       
# %%
def get_action(text_obs, session_id="0"):
    actions = {}
    for agent, raw_prompts in text_obs.items():
        # raw_prompts is a dictionary of format ({"instruction": instruction_prompt, "obs": observation_prompt})
        actions[agent] = 0

        agent_session_id = f"{agent}_{session_id}"
        response = ask_agent(agent_session_id, raw_prompts["obs"], raw_prompts["instruction"])

        match = re.search(r"action:\s*(.*)", response.content, re.IGNORECASE)
        if match:
            action_str = match.group(1).strip()
            action_enum = action_map.get(action_str)
            if action_enum:
                print(f"LLM chose action: {action_enum.name} ({action_enum.value})")
                actions[agent] = action_enum.value
            else:
                print(f"Unrecognized action: {action_str}")
        else:
            print("No valid action found in LLM response.")
        pass
    return actions 

# %%
# Test single action:
import pickle
with open("/app/Craftax/craftax/captures/0006/rendered_data.pkl", "rb") as f:
    example_obs = pickle.load(f)
returned_actions = get_action(example_obs)
print(returned_actions)

# %%
# %%
# ── Save Agent 0's chat history to disk after each step ───────────────────────
from pathlib import Path
import json

def _serialize_history(hist: InMemoryChatMessageHistory):
    out = []
    for m in hist.messages:
        # m.type is "system" | "human" | "ai" in LangChain messages
        out.append({"role": m.type, "content": m.content})
    return out

def write_agent0_history(session_id: str, dirpath: str = "."):
    """
    Writes the *entire* Agent 0 message history to a file named:
      history_agent_0_{session_id}
    The file is overwritten each call so it always reflects the latest state.
    """
    agent_id = f"agent_0_{session_id}"
    hist = histories.get(agent_id)  # defaultdict gives it if it exists
    if hist is None or len(hist.messages) == 0:
        return  # nothing to write yet

    payload = {
        "agent_id": agent_id,
        "session_id": session_id,
        "num_messages": len(hist.messages),
        "messages": _serialize_history(hist),
    }

    Path(dirpath).mkdir(parents=True, exist_ok=True)
    fname = Path(dirpath) / f"{MODEL_NAME}-history_agent_0_{session_id}"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# %%
import random
session_id = str(random.randint(0, 999999))
print("Session ID:", session_id)

total_return = 0
obs, states = env.reset(rng)

# %%
for i in range(i, 400):
    text_obs = render_craftax_text(states, env.agents)
    actions = get_action(text_obs, session_id=session_id)

    # ⬇️ update Agent 0 history file after every step
    write_agent0_history(session_id)

    rng, rng_step = jax.random.split(rng)
    obs, states, rewards, dones, infos = env.step(rng_step, states, actions)
    total_return += rewards["agent_0"]
    print("Step", i)
    print("Rewards:", rewards["agent_0"])
    if dones["agent_0"]:
        print("Episode finished.")
        print("Total return:", total_return)
        break


# %%
def peek_history(agent_id, n=3):
    hist = histories[agent_id]
    print(f"[{agent_id}] total msgs:", len(hist.messages))
    for m in hist.messages[-n:]:
        print(m.type.upper() + ":", m.content)
peek_history(f"agent_1_{session_id}", 6)

# %%
