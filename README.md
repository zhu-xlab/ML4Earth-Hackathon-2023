# ML4EARTH HACKATHON DATASET
![Alt text](https://github.com/zhu-xlab/ML4Earth-Hackathon-2023/blob/main/Source/image%20(2).png)
Large-scale hydrodynamic models generally rely on fixed-resolution spatial grids and model parameters as well as incurring a high computational cost. This limits their ability to forecast flood crests and issue time-critical hazard warnings accurately. In this task, we should build a fast, stable, accurate flood modeling framework that can perform at scales from the large scale. Specifically, we will provide the input data and ground truth data in Pakistan flood 2022. Supervised or unsupervised methods based on machine learning methods should be designed to solve the 2-D shallow water equations. Finally, based on this model, a flood forecast model should be achieved in the event of the Pakistan flood in 2022.
# Important Links
The hackathon rules are given [here](https://ml4earth23.devpost.com/)  
The starter pack notebook is given [here](https://ml4earth23.devpost.com/)  
The training dataset can be downloaded [here](https://ml4earth23.devpost.com/) 
The validation dataset can be downloaded [here](https://ml4earth23.devpost.com/) 
The data reading functions are given under folder [Code](https://ml4earth23.devpost.com/)  where:

# Study Region
![Location of the study area and elevation information](https://github.com/zhu-xlab/ML4Earth-Hackathon-2023/blob/main/Source/Picture1.png)
In the summer monsoon season of 2022, Pakistan experienced a devastating flood event. This flood event impacted approximately one-third of Pakistan's vast population, resulting in the displacement of around 32 million individuals and tragically causing the loss of 1,486 lives, including 530 children. The economic toll of this disaster has been estimated at exceeding $30 billion. The study area encompasses the regions in Pakistan most severely affected by the flood. The Indus River basin, a critical drainage system, plays a pivotal role in this study area's hydrology. 

# Data Requirements
Data include topographical data, gridded Manning coefficient, and real-time gridded rainfall data.
## Topographical data
Specifically, a high-resolution (30 m) forest and buildings removed Copernicus digital elevation model (FABDEM)} from COPDEM30  is required for flood simulation. A bilinear interpolation technique is implemented to downsample the DEM by a factor of 16 (480m), thus producing a coarse grid for the training of the model. 

##  Gridded Manning coefficient
Land cover information is useful for estimating and adjusting friction and infiltration coefficients in Floodcast. Land cover information in the study area can be subtracted from a publicly available Globa-Land30 dataset~\cite{chen2015global} developed by the Ministry of Natural Resources of China, which is shown in Fig.~\ref{fig:14} (a). It is a parcel‐based land cover map created by classifying satellite data into 8 classes, available at a spatial resolution of up to 30 m for the study area. Cultivated land is the predominant land cover type (57.05\%) in the Pakistan study area, and urban areas only account for 0.8\% of the total study area.

##  Rainfall data
The rainfall data is a grid‐based data set at $0.1^{\circ} \times 0.1^{\circ}$ spatial resolution and half-hourly temporal resolution from GPM-IMERG. Utilizing the proposed real-time rainfall processing and analysis tool in [2], rainfall data with a temporal resolution of 5 minutes (300 s) and a spatial resolution of $480 m \times 480 m$ is obtained. 

##  Initial Conditions

## Training and validation Format

# Terms of Use

# References
