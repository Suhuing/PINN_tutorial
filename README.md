# [This repository was created for the PINN tutorial]

Developed for the PINN tutorial used in the Fall 2025 course.

## Overview
이 저장소는 2025 2학기에 개설된 이동로봇 수업에 과제에 필요한 PINN Tutorial을 제작한 저장소입니다.
## Dependencies
- Python 3.10 / C++17
- Conda (Anaconda / Miniconda)

설치 예시:
```bash
conda env create -f pinn_tutorial.yml
conda activate pinn_tutorial
```

## Examples
1. **PINN MPC**
   ```bash
   cd PINN_tutorial/PINN_MPC/src
   python3 main.py
   ```
2. **Example 1**
   ```bash
   jupyter-notebook
   enter temp_pred.ipynb
   ```

3. **Example 2**
   ```bash
   jupyter-notebook
   enter pinn_vs_data.ipynb
   ```

## Acknowledgements

Part of the implementation in this repository references and adapts the open-source project
PINNs-based-MPC by Jonas Nicodemus:
https://github.com/Jonas-Nicodemus/PINNs-based-MPC

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Maintainer: [박수환] (<tnghks0605@khu.ac.kr>)  
Lab: [RCI Lab @ Kyung Hee University](https://rcilab.khu.ac.kr)
