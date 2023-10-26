# ctransformers should be installed>    !pip install ctransformers[cuda]

from ctransformers import AutoModelForCausalLM
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type = "mistral", gpu_layers = 50, context_length=2048)

Content = """<s>[INST] Summarize the key points:
In common usage, climate change describes global warming—the ongoing increase in global average temperature—and its effects on Earth's climate system. Climate change in a broader sense also includes previous long-term changes to Earth's climate. The current rise in global average temperature is more rapid than previous changes, and is primarily caused by humans burning fossil fuels.[2][3] Fossil fuel use, deforestation, and some agricultural and industrial practices add to greenhouse gases, notably carbon dioxide and methane. Greenhouse gases absorb some of the heat that the Earth radiates after it warms from sunlight. Larger amounts of these gases trap more heat in Earth's lower atmosphere, causing global warming.

Climate change is causing a range of increasing impacts on the environment. Deserts are expanding, while heat waves and wildfires are becoming more common. Amplified warming in the Arctic has contributed to melting permafrost, glacial retreat and sea ice loss. Higher temperatures are also causing more intense storms, droughts, and other weather extremes. Rapid environmental change in mountains, coral reefs, and the Arctic is forcing many species to relocate or become extinct. Even if efforts to minimise future warming are successful, some effects will continue for centuries. These include ocean heating, ocean acidification and sea level rise.
 [/INST]"""

print(llm(Content))