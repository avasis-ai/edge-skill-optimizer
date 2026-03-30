<div align="center">

<img src="https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/License-Apache--2.0-4CC61E?style=flat-square&logo=osi&logoColor=white" alt="License">
<img src="https://img.shields.io/badge/Version-0.1.0-3B82F6?style=flat-square" alt="Version">
<img src="https://img.shields.io/badge/PRs_Welcome-3B82F6?style=flat-square" alt="PRs Welcome">

<br/>
<br/>

<h3>Shrink complex agent skills to run natively on mobile devices.</h3>

<i>This pipeline takes a complex SKILL.md workflow and distills it into a highly quantized, task-specific micro-model (e.g., 1B parameters). This specialized model runs flawlessly on iOS or Android NPUs without requiring cloud connectivity.</i>

<br/>
<br/>

<a href="#installation"><b>Install</b></a>
&ensp;·&ensp;
<a href="#quick-start"><b>Quick Start</b></a>
&ensp;·&ensp;
<a href="#features"><b>Features</b></a>
&ensp;·&ensp;
<a href="#architecture"><b>Architecture</b></a>
&ensp;·&ensp;
<a href="#demo"><b>Demo</b></a>

</div>

---
## Installation

```bash
pip install edge-skill-optimizer
```

## Quick Start

```bash
edge-skill-optimizer --help
```

## Architecture

```
edge-skill-optimizer/
├── pyproject.toml
├── README.md
├── src/
│   └── edge_skill_optimizer/
│       ├── __init__.py
│       └── cli.py
├── tests/
│   └── test_edge_skill_optimizer.py
└── AGENTS.md
```

## Demo

<!-- Add screenshot or GIF here -->

> Coming soon

## Development

```bash
git clone https://github.com/avasis-ai/edge-skill-optimizer
cd edge-skill-optimizer
pip install -e .
pytest tests/ -v
```

## Links

- **Repository**: https://github.com/avasis-ai/edge-skill-optimizer
- **PyPI**: https://pypi.org/project/edge-skill-optimizer
- **Issues**: https://github.com/avasis-ai/edge-skill-optimizer/issues

---

<div align="center">
<i>Part of the <a href="https://github.com/avasis-ai">AVASIS AI</a> open-source ecosystem</i>
</div>
