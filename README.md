<p>
<div align="center">

# ML1 | AdaBoost Classifier Analysis & Optimization
</div>
</p>

<p align="center" width="100%">
    <img src="./AdaBoost/Assets/Boosting.gif" width="60%" height="60%" />
</p>

<div align="center">
    <a>
        <img src="https://img.shields.io/badge/Made%20with-Jupyter-white?style=for-the-badge&logo=Jupyter&logoColor=white">
    </a>
</div>

<br/>

<div align="center">
    <a href="https://github.com/EstevesX10/ML1-AdaBoost-Analysis-Optimization/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/EstevesX10/ML1-AdaBoost-Analysis-Optimization?style=flat&logo=gitbook&logoColor=white&label=License&color=white">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/repo-size/EstevesX10/ML1-AdaBoost-Analysis-Optimization?style=flat&logo=googlecloudstorage&logoColor=white&logoSize=auto&label=Repository%20Size&color=white">
    </a>
    <a href="">
        <img src="https://img.shields.io/github/stars/EstevesX10/ML1-AdaBoost-Analysis-Optimization?style=flat&logo=adafruit&logoColor=white&logoSize=auto&label=Stars&color=white">
    </a>
    <a href="https://github.com/EstevesX10/ML1-AdaBoost-Analysis-Optimization/blob/main/DEPENDENCIES.md">
        <img src="https://img.shields.io/badge/Dependencies-DEPENDENCIES.md-white?style=flat&logo=anaconda&logoColor=white&logoSize=auto&color=white"> 
    </a>
</div>

## Project Overview

In recent years, the rise of **Artificial Intelligence** and the widespread use of **Machine Learning** have revolutionized the way we tackle complex real-world challenges. However, due to the **diverse nature of data involved**, choosing the right algorithm is crucial to achieve efficient and effective solutions. Therefore, understanding the **strengths** and **weaknesses** behind different Machine Learning algorithms, and knowing how to **adapt them** to meet specific challenges, can become a fulcral skill to develop.

Furthermore, since the **choice of algorithm** greatly depends on the specific task and data involved, it's clear that there is no **"Master Algorithm"** (No algorithm can solve every problem). For example, while Linear Discriminants effectively delineate boundaries in data that is linearly separable, they struggle to capture relationships in more complex, higher-dimensional spaces.

This Project focuses on the following topic:

<div align="center">

> With no Master Algorithm, is it possible to improve a existing Machine Learning Algorithm in characteristics it struggles the most?
</div>

Therefore, after choosing a **Machine Learning Algorithm** and gaining a thorough understanding of its theoretical and empirical aspects, we aim to **refine it**, specifically **targeting its weaknesses** in solving classification problems.

<p align="center" width="100%">
    <img src="./AdaBoost/Assets/ThoughtProcess.png" width="45%" height="45%" />
</p>

## Classifier Selection

Nowadays, since **singular Machine Learning Algorithms** can fall short to predict the whole data given, we decided to study an **Ensemble Algorithm**. Since these Algorithms can combine outputs of multiple models it makes them more prone to **better address more complex problems** and **provide better solutions**.

Consequently, after careful consideration, we decided to focus on enhancing the **AdaBoost Algorithm M1**, which is employed in **binary classification problems**.

<table width="100%">
  <tr>
    <td width="45%">
        <div align="center">
        <b>AdaBoost</b> (Adaptive Boosting) is a type of ensemble learning technique used in machine learning to solve both <b>classification</b> and <b>regression</b> problems. It consists on training a <b>series of weak classifiers</b> on the dataset. Therefore, with each iteration, the algorithm <b>increases the focus</b> on data points that were <b>previously predicted incorrectly</b>.
        </div>
    </td>
    <td width="55%">
        <p align="center"><img src="./AdaBoost/Assets/AdaBoost_Overview.jpeg" width="100%" height="auto"/>
        </p>
    </td>
  </tr>
</table>

As a result, the AdaBoost algorithm builds a model by considering all the individual **weak classifiers** which are **weighted based on their performance**. Consequently, classifiers with **higher predictive accuracy contribute more to the final decision** which **reduces the influence of less accurate ones** in the final prediction. 

## Authorship

- **Authors** &#8594; [Gonçalo Esteves](https://github.com/EstevesX10) and [Nuno Gomes](https://github.com/NightF0x26)
- **Course** &#8594; Machine Learning I [CC2008]
- **University** &#8594; Faculty of Sciences, University of Porto

<div align="right">
<sub>
<!-- <sup></sup> -->

`README.md by Gonçalo Esteves`
</sub>
</div>