# VIC

This python program describes the use of the VIC method described at https://www.sciencedirect.com/science/article/abs/pii/S0950705118300091

To run the program the sklearn library is needed.

In this method a dataset is read and their classes are defined based by a split point. 
The splits points can be defined in line 187 along with the number of threads the user desires to run with, defaulting to 2, and being 5 in this example:

```
187: splits = [100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255,260,265,270,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345]
188: 
189: clstrs = VIC(loginL.values,splits,30,5)
```


Multiple classifiers can also be used an defined in line 85. 

```
85: clfs = [SVC(),RandomForestClassifier(),GaussianNB(),LinearDiscriminantAnalysis(),MLPClassifier()]
```

The classifiers follow the structure of sklearn. If a new model is being integrated the BNHandler class can be used to follow this structure.

The VIC can be run with multiple threads. Depending on the threads the split datasets are organized. While implementing a tensorflow model this causes an error, due to the session handling. One workaround is to use the GPU devices as different threads and apply the different classifiers in the BNHandler class.

