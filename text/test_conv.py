# %%
import os
import sys
sys.path.append('/app/Craftax/craftax')
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"

from craftax_marl.renderer.renderer_text import render_craftax_text

# %%
import pickle
capture_dir = '/app/Craftax/craftax/captures/0006'
with open(f'{capture_dir}/state.pkl', 'rb') as f:
    state = pickle.load(f)

# Render the text data
rendered_data = render_craftax_text(state, ["agent_0", "agent_1", "agent_2"])

# Create directories and save data
with open(f'{capture_dir}/rendered_data.pkl', 'wb') as f:
    pickle.dump(rendered_data, f)
for agent, data in rendered_data.items():
    agent_dir = os.path.join(capture_dir, agent)
    os.makedirs(agent_dir, exist_ok=True)
    
    # Save instruction prompt
    with open(os.path.join(agent_dir, "instruction.txt"), "w") as instruction_file:
        instruction_file.write(data["instruction"])
    
    # Save observation prompt
    with open(os.path.join(agent_dir, "obs.txt"), "w") as obs_file:
        obs_file.write(data["obs"])

# %%
