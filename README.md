# ML4EARTH HACKATHON DATASET
![Alt text](https://github.com/zhu-xlab/ML4Earth-Hackathon-2023/blob/main/Source/image%20(2).png)
Large-scale hydrodynamic models generally rely on fixed-resolution spatial grids and model parameters as well as incurring a high computational cost. This limits their ability to forecast flood crests and issue time-critical hazard warnings accurately. In this task, we should build a fast, stable, accurate flood modeling framework that can perform at scales from the large scale. Specifically, we will provide the input data and ground truth data in Pakistan flood 2022. Supervised or unsupervised methods based on machine learning methods should be designed to solve the 2-D shallow water equations. Finally, based on this model, a flood forecast model should be achieved in the event of the Pakistan flood in 2022.

# Specific tasks for ML4Earth Hackathon  
Can we accurately predict the flood extents and depths within a 12-hour timeframe from 0:00-12:00 on August 18, 2022 (Time step=30s)?



# Important Links
The hackathon rules are given [here](https://ml4earth23.devpost.com/)  
## Input Data
Topographical data are given [here](https://drive.google.com/drive/folders/1X7ZmEvx1KUwSlLCli47UYk9bxgrbLpmo?usp=drive_link)   
Gridded Manning coefficients are given [here](https://drive.google.com/drive/folders/1X7ZmEvx1KUwSlLCli47UYk9bxgrbLpmo?usp=drive_link)   
Initial Conditions (initial water depth) are given [here](https://drive.google.com/drive/folders/1X7ZmEvx1KUwSlLCli47UYk9bxgrbLpmo?usp=drive_link)    
Rainfall data are given given [here](https://drive.google.com/drive/folders/1CF3nQcfJQ2zs2yUtnsEY3LTrT4PyzrPg?usp=drive_link)  
## Training and validation Dataset 
The training dataset can be downloaded  [here](https://drive.google.com/drive/folders/1pe5x6Nz1B6COlfE4j4YTefCe7SCouIKP?usp=drive_lin)  
The validation dataset can be downloaded [here](https://drive.google.com/drive/folders/1ygBN8rgSAoUpdFADgRAc0UQEM3FJS2vs?usp=drive_link) 
## How to read these data
The data reading functions are given under folder [Code/dataset.py](https://ml4earth23.devpost.com/)  
Complete codes of the benchmark model can be downloaded [Code/Benchmark](https://ml4earth23.devpost.com/)  

# Study Region
![Location of the study area and elevation information](https://github.com/zhu-xlab/ML4Earth-Hackathon-2023/blob/main/Source/Picture1.png)  
In the summer monsoon season of 2022, Pakistan experienced a devastating flood event. This flood event impacted approximately one-third of Pakistan's vast population, resulting in the displacement of around 32 million individuals and tragically causing the loss of 1,486 lives, including 530 children. The economic toll of this disaster has been estimated at exceeding $30 billion. The study area encompasses the regions in Pakistan most severely affected by the flood. The Indus River basin, a critical drainage system, plays a pivotal role in this study area's hydrology. 

# Input Data
The inputs to a hydraulic simulation include an elevation map, initial conditions,  boundary conditions, and the rainfall conditions in the Pakistan study region. 

## Topographical data
A high-resolution (30 m) forest and buildings removed Copernicus digital elevation model from COPDEM30  is required for flood simulation. A bilinear interpolation technique is implemented to downsample the DEM by a factor of 16 (480m). 

##  Gridded Manning coefficient
Land cover information is useful for estimating and adjusting friction coefficients in Floodcast. 

##  Initial Conditions
The flood inundation depth on 0:00 August 18 is extracted using FABDEM and the SAR-based flood extent, based on the tool in [1,3]

##  Rainfall data
The rainfall data is a grid‐based data set at $0.1^{\circ} \times 0.1^{\circ}$ spatial resolution and half-hourly temporal resolution from GPM-IMERG. Utilizing the proposed real-time rainfall processing and analysis tool in [2], rainfall data with a temporal resolution of 30s and a spatial resolution of $480 m \times 480 m$ is obtained. 

## Boundary Conditions
If boundary conditions are to be considered in your designed model, the study area only considers the discharges at the inflow boundary, which is 13236m3/s.

# Training and validation Dataset
In a 12-hour timeframe (43,200 seconds), we split the dataset into a training set comprising the first 30,240 seconds (1,008 dynamic process results) and a validation set comprising the remaining 12,960 seconds (432 dynamic process results). This division facilitates effective model training and evaluation.

# Terms of Use
The data used in this study is sourced from a paper under review at TUM AI4EO. For the hackathon, a partial dataset has been made available. The comprehensive Pakistan evolving benchmark dataset for flood prediction will be released once the paper is accepted. Kindly use it only for the purpose for which it was provided.

# References
[1] Xu, Q., Shi, Y., Guo J., Ouyang, C., & Zhu, X. X. (2023). UCDFormer: Unsupervised Change Detection Using a Transformer-driven Image Translation. IEEE Transactions on Geoscience and Remote Sensing.  
[2] Xu, Q., Shi, Y., Bamber, J., Ouyang, C., & Zhu, X. X. (2023). A large-scale flood modeling using geometry-adaptive physics-informed neural solver and Earth observation data (No. EGU23-3276).  
[3] Xu, Q., Shi, Y., Bamber, J., Ouyang, C., & Zhu, X. X. (2023). Large-scale flood modeling and forecasting with FloodCast. 

