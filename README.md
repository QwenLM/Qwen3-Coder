<a name="readme-top"></a>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/qwen3_coder.png" width="400"/>
<p>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/qwen3-coder-main.jpg" width="800"/>
<p>

<p align="center">
        💜 <a href="https://chat.qwenlm.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen/models">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen3-coder">Blog</a> &nbsp&nbsp ｜ &nbsp&nbsp📖 <a href="https://qwen.readthedocs.io/">Documentation</a>
<br> 
</a>&nbsp&nbsp | &nbsp&nbsp 🌍 <a href="https://huggingface.co/spaces/Qwen/Qwen3-Coder-WebDev">WebDev</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD"> Discord</a>&nbsp&nbsp | &nbsp&nbsp 📄 <a href="https://arxiv.org/abs/2505.09388">Arxiv</a>&nbsp&nbsp | &nbsp&nbsp 👽 <a href="https://github.com/QwenLM/qwen-code">Qwen Code</a>
</p>

Visit our Hugging Face or ModelScope organization (click links above), search checkpoints with names starting with `Qwen3-Coder-`, and you will find all you need! Enjoy!

# Qwen3-Coder: Agentic Coding in the World.

## Introduction

Today, we're announcing Qwen3-Coder, our most agentic code model to date. **Qwen3-Coder** is available in multiple sizes, but we're excited to introduce its most powerful variant first: **Qwen3-Coder-480B-A35B-Instruct** — a 480B-parameter Mixture-of-Experts model with 35B active parameters, offering exceptional performance in both coding and agentic tasks. **Qwen3-Coder-480B-A35B-Instruct** sets new state-of-the-art results among open models on Agentic Coding, Agentic Browser-Use, and Agentic Tool-Use, comparable to Claude Sonnet. 

💻 **Significant Performance**: among open models on **Agentic Coding**, **Agentic Browser-Use**, and other foundational coding tasks, achieving results comparable to Claude Sonnet;

📚 **Long-context Capabilities**: with native support for **256K** tokens, extendable up to **1M** tokens using Yarn, optimized for repository-scale understanding;

🛠 **Agentic Coding**: supporting for most platform such as **Qwen Code**, **CLINE**, featuring a specially designed function call format;

## Basic information

1. ✨ Supporting long context understanding and generation with the context length of 256K tokens;
2. ✨ Supporting 358 coding languages;
```
['ABAP', 'ActionScript', 'Ada', 'Agda', 'Alloy', 'ApacheConf', 'AppleScript', 'Arc', 'Arduino', 'AsciiDoc', 'AspectJ', 'Assembly', 'Augeas', 'AutoHotkey', 'AutoIt', 'Awk', 'Batchfile', 'Befunge', 'Bison', 'BitBake', 'BlitzBasic', 'BlitzMax', 'Bluespec', 'Boo', 'Brainfuck', 'Brightscript', 'Bro', 'C', 'C#', 'C++', 'C2hs Haskell', 'CLIPS', 'CMake', 'COBOL', 'CSS', 'CSV', "Cap'n Proto", 'CartoCSS', 'Ceylon', 'Chapel', 'ChucK', 'Cirru', 'Clarion', 'Clean', 'Click', 'Clojure', 'CoffeeScript', 'ColdFusion', 'ColdFusion CFC', 'Common Lisp', 'Component Pascal', 'Coq', 'Creole', 'Crystal', 'Csound', 'Cucumber', 'Cuda', 'Cycript', 'Cython', 'D', 'DIGITAL Command Language', 'DM', 'DNS Zone', 'Darcs Patch', 'Dart', 'Diff', 'Dockerfile', 'Dogescript', 'Dylan', 'E', 'ECL', 'Eagle', 'Ecere Projects', 'Eiffel', 'Elixir', 'Elm', 'Emacs Lisp', 'EmberScript', 'Erlang', 'F#', 'FLUX', 'FORTRAN', 'Factor', 'Fancy', 'Fantom', 'Forth', 'FreeMarker', 'G-code', 'GAMS', 'GAP', 'GAS', 'GDScript', 'GLSL', 'Genshi', 'Gentoo Ebuild', 'Gentoo Eclass', 'Gettext Catalog', 'Glyph', 'Gnuplot', 'Go', 'Golo', 'Gosu', 'Grace', 'Gradle', 'Grammatical Framework', 'GraphQL', 'Graphviz (DOT)', 'Groff', 'Groovy', 'Groovy Server Pages', 'HCL', 'HLSL', 'HTML', 'HTML+Django', 'HTML+EEX', 'HTML+ERB', 'HTML+PHP', 'HTTP', 'Haml', 'Handlebars', 'Harbour', 'Haskell', 'Haxe', 'Hy', 'IDL', 'IGOR Pro', 'INI', 'IRC log', 'Idris', 'Inform 7', 'Inno Setup', 'Io', 'Ioke', 'Isabelle', 'J', 'JFlex', 'JSON', 'JSON5', 'JSONLD', 'JSONiq', 'JSX', 'Jade', 'Jasmin', 'Java', 'Java Server Pages', 'JavaScript', 'Julia', 'Jupyter Notebook', 'KRL', 'KiCad', 'Kit', 'Kotlin', 'LFE', 'LLVM', 'LOLCODE', 'LSL', 'LabVIEW', 'Lasso', 'Latte', 'Lean', 'Less', 'Lex', 'LilyPond', 'Linker Script', 'Liquid', 'Literate Agda', 'Literate CoffeeScript', 'Literate Haskell', 'LiveScript', 'Logos', 'Logtalk', 'LookML', 'Lua', 'M', 'M4', 'MAXScript', 'MTML', 'MUF', 'Makefile', 'Mako', 'Maple', 'Markdown', 'Mask', 'Mathematica', 'Matlab', 'Max', 'MediaWiki', 'Metal', 'MiniD', 'Mirah', 'Modelica', 'Module Management System', 'Monkey', 'MoonScript', 'Myghty', 'NSIS', 'NetLinx', 'NetLogo', 'Nginx', 'Nimrod', 'Ninja', 'Nit', 'Nix', 'Nu', 'NumPy', 'OCaml', 'ObjDump', 'Objective-C++', 'Objective-J', 'Octave', 'Omgrofl', 'Opa', 'Opal', 'OpenCL', 'OpenEdge ABL', 'OpenSCAD', 'Org', 'Ox', 'Oxygene', 'Oz', 'PAWN', 'PHP', 'POV-Ray SDL', 'Pan', 'Papyrus', 'Parrot', 'Parrot Assembly', 'Parrot Internal Representation', 'Pascal', 'Perl', 'Perl6', 'Pickle', 'PigLatin', 'Pike', 'Pod', 'PogoScript', 'Pony', 'PostScript', 'PowerShell', 'Processing', 'Prolog', 'Propeller Spin', 'Protocol Buffer', 'Public Key', 'Pure Data', 'PureBasic', 'PureScript', 'Python', 'Python traceback', 'QML', 'QMake', 'R', 'RAML', 'RDoc', 'REALbasic', 'RHTML', 'RMarkdown', 'Racket', 'Ragel in Ruby Host', 'Raw token data', 'Rebol', 'Red', 'Redcode', "Ren'Py", 'RenderScript', 'RobotFramework', 'Rouge', 'Ruby', 'Rust', 'SAS', 'SCSS', 'SMT', 'SPARQL', 'SQF', 'SQL', 'STON', 'SVG', 'Sage', 'SaltStack', 'Sass', 'Scala', 'Scaml', 'Scheme', 'Scilab', 'Self', 'Shell', 'ShellSession', 'Shen', 'Slash', 'Slim', 'Smali', 'Smalltalk', 'Smarty', 'Solidity', 'SourcePawn', 'Squirrel', 'Stan', 'Standard ML', 'Stata', 'Stylus', 'SuperCollider', 'Swift', 'SystemVerilog', 'TOML', 'TXL', 'Tcl', 'Tcsh', 'TeX', 'Tea', 'Text', 'Textile', 'Thrift', 'Turing', 'Turtle', 'Twig', 'TypeScript', 'Unified Parallel C', 'Unity3D Asset', 'Uno', 'UnrealScript', 'UrWeb', 'VCL', 'VHDL', 'Vala', 'Verilog', 'VimL', 'Visual Basic', 'Volt', 'Vue', 'Web Ontology Language', 'WebAssembly', 'WebIDL', 'X10', 'XC', 'XML', 'XPages', 'XProc', 'XQuery', 'XS', 'XSLT', 'Xojo', 'Xtend', 'YAML', 'YANG', 'Yacc', 'Zephir', 'Zig', 'Zimpl', 'desktop', 'eC', 'edn', 'fish', 'mupad', 'nesC', 'ooc', 'reStructuredText', 'wisp', 'xBase']
```
3. ✨ Retain strengths in math and general capabilities from base model.

> [!Important]
> 
> Qwen3-coder function calling relies on our new tool parser `qwen3coder_tool_parser.py` <a href="https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct/blob/main/qwen3coder_tool_parser.py">here</a>.
>
> We updated both the special tokens and their corresponding token ids, in order to maintain consistency with Qwen3. Please make sure to use the new tokenizer.


| model name                  | type     | length | Download                                                                                                                                                                        |
|-----------------------------|----------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Qwen3-Coder-480B-A35B-Instruct         | instruct     | 256k    | 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct  ) • 🤖 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Coder-480B-A35B-Instruct)                                       |
| Qwen3-Coder-480B-A35B-Instruct-FP8         | instruct     | 256k    | 🤗 [Hugging Face](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) • 🤖 [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8)                                       |

Detailed performance and introduction are shown in this <a href="https://qwenlm.github.io/blog/qwen3-coder"> 📑 blog</a>.

## Quick Start

> [!Important]
> **Qwen3-Coder-480B-A35B-Instruct** are instruction models for chatting;
>
> This model supports only non-thinking mode and does not generate ``<think></think>`` blocks in its output. Meanwhile, specifying `enable_thinking=False` is no longer required.**
>
### 👉🏻 Chat with Qwen3-Coder-480B-A35B-Instruct
You can just write several lines of code with `transformers` to chat with Qwen3-Coder-480B-A35B-Instruct. Essentially, we build the tokenizer and the model with `from_pretrained` method, and we use generate method to perform chatting with the help of chat template provided by the tokenizer. Below is an example of how to chat with **Qwen3-Coder-480B-A35B-Instruct**:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "write a quick sort algorithm."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=65536
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
The `apply_chat_template()` function is used to convert the messages into a format that the model can understand.
The `add_generation_prompt` argument is used to add a generation prompt, which refers to `<|im_start|>assistant\n` to the input. Notably, we apply ChatML template for chat models following our previous practice.
The `max_new_tokens` argument is used to set the maximum length of the response. The `tokenizer.batch_decode()` function is used to decode the response. In terms of the input, the above messages is an example to show how to format your dialog history and system prompt.
You can use the other size of instruct model in the same way.


#### Fill in the middle with Qwen3-Coder-480B-A35B-Instruct

The code insertion task, also referred to as the "fill-in-the-middle" challenge, requires the insertion of code segments in a manner that bridges the gaps within a given code context. For an approach aligned with best practices, we recommend adhering to the formatting guidelines outlined in the paper "Efficient Training of Language Models to Fill in the Middle"[[arxiv](https://arxiv.org/abs/2207.14255)]. 

The prompt should be structured as follows:
```python
prompt = '<|fim_prefix|>' + prefix_code + '<|fim_suffix|>' + suffix_code + '<|fim_middle|>'
```
Following the approach mentioned, an example would be structured in this manner:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
# load model
device = "cuda" # the device to load the model onto

TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-480B-A35B-Instruct")
MODEL = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Coder-480B-A35B-Instruct", device_map="auto").eval()


input_text = """<|fim_prefix|>def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    <|fim_suffix|>
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)<|fim_middle|>"""
            
messages = [
    {"role": "system", "content": "You are a code completion assistant."},
    {"role": "user", "content": input_text}
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = TOKENIZER([text], return_tensors="pt").to(model.device)

# Use `max_new_tokens` to control the maximum output length.
generated_ids = MODEL.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=False)[0]
# The generated_ids include prompt_ids, we only need to decode the tokens after prompt_ids.
output_text = TOKENIZER.decode(generated_ids[len(model_inputs.input_ids[0]):], skip_special_tokens=True)

print(f"Prompt: {input_text}\n\nGenerated text: {output_text}")
```

## Use Cases
### Example: Physics-Based Chimney Demolition Simulation with Controlled Explosion

<details>
<summary>Prompt with Qwen Chat Web Dev </summary>

```
使用 three.js, cannon-es.js 生成一个震撼的3D建筑拆除演示。

## 场景设置：
- 地面是一个深灰色混凝土平面，尺寸80*80，
- 所有物体严格遵循现实物理规则，包括重力、摩擦力、碰撞检测和动量守恒

## 建筑结构：
- 一座圆形高层建筑，周长对应20个方块
- 建筑总高度60个方块
- 每层采用砖砌结构，方块与砖结构建筑一致, 错开50%排列，增强结构稳定性
- 建筑外墙使用米色方块
- **重要：方块初始排列时必须确保紧密贴合，无间隙，可以通过轻微重叠或调整半径来实现**
- **重要：建筑初始化完成后，所有方块应该处于物理"睡眠"状态，确保建筑在爆炸前保持完美的静止状态，不会因重力而下沉或松散**
- 建筑砖块之间使用粘性材料填充（不可见），通过高摩擦力（0.8+）和低弹性（0.05以下）来模拟粘合效果
- 砖块在建筑倒塌瞬间不会散掉，而是建筑作为一个整体倒在地面的时候才因受力过大而散掉

## 定向爆破系统：
- 在建筑的第1层的最右侧方块附近安装爆炸装置（不可见）
- 提供操作按钮点击爆炸
- **爆炸时唤醒所有相关方块的物理状态**
- 爆炸点产生半径2的强力冲击波，冲击波影响到的方块, 受到2-5单位的冲击力

## 建筑稳定性要求：
- **确保建筑在未爆炸时完全静止，无任何晃动或下沉**
- **物理世界初始化后给建筑几个物理步骤来自然稳定，或使用睡眠机制**
- **方块间的接触材料应具有高摩擦力和极低弹性，模拟砖块间的砂浆粘合**

## 震撼的倒塌效果：
- 方块在爆炸冲击下不仅飞散，还会在空中翻滚和碰撞
- 烟尘会随着建筑倒塌逐渐扩散，营造真实的拆除现场氛围

## 增强的视觉效果：
- 添加环境光照变化：爆炸瞬间亮度激增，然后被烟尘遮挡变暗
- 粒子系统包括：烟雾、灰尘

## 技术要求：
- 粒子系统用于烟雾和灰尘效果
- 所有代码集成在单个HTML文件中，包含必要的CSS样式
- 添加简单的UI控制：重置按钮、相机角度切换, 爆炸按钮, 鼠标左键控制摄像机角度，右键控制摄像机位置，滚轮控制摄像机焦距
```

</details>

<p align="center">
    <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/demo1.mp4">
    <img src="assets/usage_demo_example1.png" width="400" />
    </a>
<p >

### Example: Multicolor and Interactive Animation

<details>
<summary>Prompt with Cline [act mode] </summary>

```
Create an amazing animation multicolor and interactive using p5js

use this cdn:
https://cdn.jsdelivr.net/npm/p5@1.7.0/lib/p5.min.js
```
</details>

<p align="center">
    <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/demo2.mp4">
    <img src="assets/usage_demo_example2.png" width="400" />
    </a>
<p >

### Example: 3D Google Earth

<details>
<summary>Prompt with Qwen Chat Web Dev </summary>

```
To create a 3D Google Earth, you need to load the terrain map correctly. You can use any online resource. The code is written into an HTML file.
```

</details>

<p align="center">
    <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/demo3.mp4">
    <img src="assets/usage_demo_example3.png" width="400" />
    </a>
<p >

### Example: Testing Your WPM with a Famous Quote 


<details>
<summary> Prompt with Qwen-Code CLI </summary>

```
Create an interesting typing game with a keyboard in the lower middle of the screen and some famous articles in the upper middle. When the user types a word correctly, a cool reaction should be given to encourage him. Design a modern soft color scheme inspired by macarons. Come up with a very creative solution first, and then start writing code.
The game should be able to support typing, and you need to neglect upcase and lowercase.
```
</details>

<p align="center">
    <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/demo4.mp4">
    <img src="assets/usage_demo_example4.png" width="400" />
    </a>
<p >

### Example: Bouncing Ball in Rotation Hypercube


<details>
<summary> Prompt with Qwen Chat Web Dev </summary>

```
Make a page in HTML that shows an animation of a ball bouncing in a rotating hypercube
```
</details>

<p align="center">
    <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/demo5.mp4">
    <img src="assets/usage_demo_example5.png" width="400" />
    </a>
<p >

### Example: Solar System Simulation


<details>
<summary> Prompt with Cline [act mode] </summary>

```
write a web page to show the solar system simulation
```
</details>


<p align="center">
    <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/demo6.mp4">
    <img src="assets/usage_demo_example6.png" width="400" />
    </a>
<p >

### Example: DUET Game


<details>
<summary> Prompt with Cline [act mode] </summary>

```
Create a complete, single-file HTML game with CSS and JavaScript. The game is inspired by "Duet".

Gameplay:

There are two balls, one red and one blue, rotating around a central point.
The player uses the 'A' and 'D' keys to rotate them counter-clockwise and clockwise.
White rectangular obstacles move down from the top of the screen.
The player must rotate the balls to avoid hitting the obstacles.
If a ball hits an obstacle, the game is over.
Visuals:

Make the visual effects amazing.
Use a dark background with neon glowing effects for the balls and obstacles.
Animations should be very smooth.
```
</details>


<p align="center">
    <a href="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Coder/demo7.mp4">
    <img src="assets/usage_demo_example7.png" width="400" />
    </a>
<p >


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen3-Coder&type=Date)](https://star-history.com/#QwenLM/Qwen3-Coder&Date)



## Citation
If you find our work helpful, feel free to give us a cite.

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388},
}
@article{hui2024qwen2,
  title={Qwen2. 5-Coder Technical Report},
  author={Hui, Binyuan and Yang, Jian and Cui, Zeyu and Yang, Jiaxi and Liu, Dayiheng and Zhang, Lei and Liu, Tianyu and Zhang, Jiajun and Yu, Bowen and Dang, Kai and others},
  journal={arXiv preprint arXiv:2409.12186},
  year={2024}
}
```

## Contact Us
If you are interested to leave a message to either our research team or product team, join our [Discord](https://discord.gg/z3GAxXZ9Ce) or [WeChat groups](https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png)!

<p align="right" style="font-size: 14px; color: #555; margin-top: 20px;">
    <a href="#readme-top" style="text-decoration: none; color: #007bff; font-weight: bold;">
        ↑ Back to Top ↑
    </a>
</p>
