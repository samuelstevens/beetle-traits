Goal of this experiment is to see if there is any guidance that can be given by training models on different amounts of annotations per genus, from a few samples up to a group image with 20+ annotations. 

The prediction is that adding more annotations for a genus will decrease the percent error length of the validation inference, scaled by the amount of annotations added. I am interested in seeing if adding a few annotations for a genus has an outsized impact on the percent error length
Genura to target: Metrius (n: 47), Discoderus (n: 79), Dicheirus: (n: 57)

One tray image from each genura was annotated. There will be 5 training runs: 0 samples per genura, 5, 10, 15, and all. I don't think it is good experimental design to not split up the training runs further where one run only has 5 samples from one genus, but I don't think that will have an impact on outsized impact if it exists. 

These samples are going to training of BioRepo, editing the parameter of --training_annotations. All training annotations from round1 will be included.