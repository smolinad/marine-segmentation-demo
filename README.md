# Marine Segmentation Demo

A demo project for marine image segmentation using Python and Picamera2.

---

## Prerequisites

Ensure the following programs and packages are installed (already installed in the surveyor's Pi `SR1.8-019`):

| Program | Installation Command |
|---------|--------------------|
| `uv` | `curl -LsSf https://astral.sh/uv/install.sh \| sh && source $HOME/.bashrc` |
| `python3-picamera2` | `sudo apt install python3-picamera2` |

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/smolinad/marine-segmentation-demo.git
```

```bash
cd marine-segmentation-demo
```

3. Install project dependencies:

```bash
uv sync
```

---

## Usage

Run the main application:
```bash 
uv run marine.py
```
Feel free to iterate over the marine.py file to improve segmentation (add more classes, etc) :)

---


