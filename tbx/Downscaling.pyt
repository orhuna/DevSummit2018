"""
Source Name:   SVM.pyt
Version:       ArcGIS Pro
Author:        Environmental Systems Research Institute Inc.
Description:   Python tool to perform statistical climate downscaling using SVM and Ridge Regression
"""

import arcpy as ARCPY
import arcgisscripting as ARC
import numpy as NUM
import SSDataObject as SSDO
import matplotlib.pyplot as PLT
from sklearn.svm import SVR
from sklearn.linear_model import Ridge as RIDGE
from sklearn.linear_model import RidgeCV as RIDGECV
from sklearn.metrics import mean_squared_error as MSE

########################### Globals ####################################
def writeFC(inFC, outFC, data):
    ARCPY.env.overwriteOutput = True
    #import pdb; pdb.set_trace()
    ssdo  = SSDO.SSDataObject(inFC)
    ssdo.obtainData()
    fields = SSDO.CandidateField(name = 'T_predict', data = data, 
                                 type = 'DOUBLE', )
    #Write Out the Feature Class
    ARC._ss.output_featureclass_from_dataobject(ssdo, outFC, [fields])

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Support Vector Machine"
        self.alias = "SVM"
        self.helpContext = 50

        #### List of tool classes associated with this toolbox ####
        self.tools = [SVM, RidgeRegression]

    #### Run Analysis ####

class SVM(object):
    def __init__(self):
        self.label = "Support Vector Machine"
        self.decription  = "Non-Linear Regression"
        self.canRunInBackground = False
        self.helpContext = 50000005

    def getParameterInfo(self):
        """Define parameter definitions"""
        #### Local Imports ####
        import os as OS
        import sys as SYS

        templateDir = OS.path.join(OS.path.dirname(SYS.path[0]), "Templates", "Layers")
        fullRLF = OS.path.join(templateDir, "LocalColocationQuotient.lyrx")

        param0 = ARCPY.Parameter(displayName="Training Dataset",
                                 name="train_dataset",
                                 datatype="GPFeatureLayer",
                                 parameterType="Required",
                                 direction="Input")
        param0.filter.list = ["Point"]

        param1 = ARCPY.Parameter(displayName="Output Feature",
                                name="output_feature",
                               datatype="DEFeatureClass",
                              parameterType="Required",
                             direction="Output")

        param2 = ARCPY.Parameter(displayName="Predictor Variables",
                                name="predictors",
                               datatype="Field",
                              parameterType="Required",
                             direction="Input",
                            multiValue=True)
        param2.parameterDependencies = ["train_dataset"]

        param3 = ARCPY.Parameter(displayName="Prediction Variable",
                                name="predictand",
                               datatype="Field",
                              parameterType="Required",
                             direction="Input",
                            multiValue=False)
        param3.parameterDependencies = ["train_dataset"]

        param4 = ARCPY.Parameter(displayName="Prediction Dataset",
                                name="in_features1",
                               datatype="GPFeatureLayer",
                              parameterType="Required",
                             direction="Input")
        param4.filter.list = ["Point"]

        param5 = ARCPY.Parameter(displayName="Error Penalty",
                                name="C",
                               datatype="GPDouble",
                              parameterType="Required",
                             direction="Input")

        param6 = ARCPY.Parameter(displayName="Kernel", 
                                 name="kernel_type", 
                                 datatype="GPString", 
                                 parameterType="Optional", 
                                 direction="Input")
        param6.filter.type = "Value List"
        param6.filter.list = ["LINEAR", "RBF", "POLY", "SIGMOID"]
        param6.value = "RBF"

        param7 = ARCPY.Parameter(displayName="Kernel Gamma",
                                name="field",
                               datatype="GPDouble",
                              parameterType="Required",
                             direction="Input")

        params = [param0, param1, param2, param3, param4, param5, param6, param7]
        
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        #Text Inputs
        inputFC = parameters[0].valueAsText
        outputFC = parameters[1].valueAsText
        predictFC = parameters[4].valueAsText
        kernelType = parameters[6].valueAsText.lower()

        #Numeric Inputs
        C = float(parameters[5].valueAsText)
        gamma = float(parameters[7].valueAsText)

        predictVars = parameters[2].valueAsText.upper().split(';')
        predictandVar = parameters[3].valueAsText.upper()

        allVars = predictVars + [predictandVar]

        ssdo = SSDO.SSDataObject(inputFC, useChordal = False)
        ssdo.obtainData(ssdo.oidName, allVars)

        dataPredict = NUM.zeros((ssdo.numObs, len(predictVars)))

        for varNum, var in enumerate(predictVars):
            dataPredict[:,varNum] = ssdo.fields[var].data

        predictand = ssdo.fields[predictandVar].data

        ARCPY.AddMessage('{0} predictors is being used for regression'.format(len(predictVars)))
        
        #Set-up SVM Predictor
        svr_rbf = SVR(kernel=kernelType, C=C, gamma=gamma)
        y_rbf = svr_rbf.fit(dataPredict, predictand).predict(dataPredict)
        
        #Compare Observed and Predicted Data
        lw = 2
        PLT.plot(y_rbf, color='navy', lw=lw, label='Model Fit')
        PLT.plot(predictand, color='red', lw=lw, label='Station Data')
        PLT.title('Support Vector Regression Prediction')
        PLT.xlabel('Time (days)')
        PLT.ylabel('Temperature (C)')
        PLT.legend()
        PLT.show()

        ##Perform Prediction
        ssdoPred = SSDO.SSDataObject(predictFC, useChordal = False)
        ssdoPred.obtainData(ssdoPred.oidName, predictVars)

        dataPredict = NUM.zeros((ssdoPred.numObs, len(predictVars)))

        for varNum, var in enumerate(predictVars):
            dataPredict[:,varNum] = ssdoPred.fields[var].data

        y_hat_downscale = svr_rbf.predict(dataPredict)

        #writeFC(predictFC, outputFC, y_hat_downscale)

        return

class RidgeRegression(object):
    def __init__(self):
        self.label = "Ridge Regression"
        self.decription  = "Non-Linear Regression"
        self.canRunInBackground = False
        self.helpContext = 50000005

    def getParameterInfo(self):
        """Define parameter definitions"""
        #### Local Imports ####
        import os as OS
        import sys as SYS

        templateDir = OS.path.join(OS.path.dirname(SYS.path[0]), "Templates", "Layers")
        fullRLF = OS.path.join(templateDir, "LocalColocationQuotient.lyrx")

        param0 = ARCPY.Parameter(displayName="Training Dataset",
                                 name="train_dataset",
                                 datatype="GPFeatureLayer",
                                 parameterType="Required",
                                 direction="Input")
        param0.filter.list = ["Point"]

        param1 = ARCPY.Parameter(displayName="Output Feature",
                                name="output_feature",
                               datatype="DEFeatureClass",
                              parameterType="Required",
                             direction="Output")

        param2 = ARCPY.Parameter(displayName="Predictor Variables",
                                name="predictors",
                               datatype="Field",
                              parameterType="Required",
                             direction="Input",
                            multiValue=True)
        param2.parameterDependencies = ["train_dataset"]

        param3 = ARCPY.Parameter(displayName="Prediction Variable",
                                name="predictand",
                               datatype="Field",
                              parameterType="Required",
                             direction="Input",
                            multiValue=False)
        param3.parameterDependencies = ["train_dataset"]

        param4 = ARCPY.Parameter(displayName="Prediction Dataset",
                                name="in_features1",
                               datatype="GPFeatureLayer",
                              parameterType="Required",
                             direction="Input")
        param4.filter.list = ["Point"]

        param5 = ARCPY.Parameter(displayName="Fit Intercept",
                                name="intercept",
                               datatype="GPBoolean",
                              parameterType="Optional",
                             direction="Input")

        param6 = ARCPY.Parameter(displayName="Regularization Strength", 
                                 name="alpha", 
                                 datatype="GPDouble", 
                                 parameterType="Optional", 
                                 direction="Input")

        param7 = ARCPY.Parameter(displayName="Normalize",
                                name="field",
                               datatype="GPBoolean",
                              parameterType="Optional", 
                             direction="Input")

        params = [param0, param1, param2, param3, param4, param5, param6, param7]
        
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        #Text Inputs
        inputFC = parameters[0].valueAsText
        outputFC = parameters[1].valueAsText
        predictFC = parameters[4].valueAsText
        intercept = parameters[5].valueAsText
        normalize = parameters[7].valueAsText
        #Numeric Inputs
        alpha = parameters[6].valueAsText
        if alpha is not None:
            alpha = float(parameters[6].valueAsText)

        predictVars = parameters[2].valueAsText.upper().split(';')
        predictandVar = parameters[3].valueAsText.upper()

        allVars = predictVars + [predictandVar]

        ssdo = SSDO.SSDataObject(inputFC, useChordal = False)
        ssdo.obtainData(ssdo.oidName, allVars)

        dataPredict = NUM.zeros((ssdo.numObs, len(predictVars)))

        for varNum, var in enumerate(predictVars):
            dataPredict[:,varNum] = ssdo.fields[var].data

        predictand = ssdo.fields[predictandVar].data

        ARCPY.AddMessage('{0} predictors are     being used for regression'.format(len(predictVars)))
        
        if alpha is None:
            ridge = RIDGECV(alphas=NUM.logspace(-10, -2, 100), fit_intercept = intercept, normalize = normalize, store_cv_values = True)

            ridge.fit(dataPredict, predictand)
            ARCPY.AddMessage('Optimum regularization strength is {0}'.format(ridge.alpha_))
            # Display results
            ax = PLT.gca()
            ax.plot(ridge.alphas, NUM.mean(ridge.cv_values_, axis = 0))
            ax.set_xscale('log')
            ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
            PLT.xlabel('Regularization Strength')
            PLT.ylabel('CV')
            PLT.title('Fit Quality (CV) wrt Alpha')
            PLT.axis('tight')
            PLT.show()

        else:
            ridge = RIDGE(alpha=alpha, fit_intercept=intercept, normalize =  normalize)
            ridge.fit(dataPredict, predictand)

        yhat = ridge.predict(dataPredict)
        # #############################################################################
        

         #Compare Observed and Predicted Data
        lw = 2
        PLT.plot(ridge.predict(dataPredict), color='navy', lw=lw, label='Model Fit')
        PLT.plot(predictand, color='red', lw=lw, label='Station Data')
        PLT.title('Support Vector Regression Prediction')
        PLT.xlabel('Time (days)')
        PLT.ylabel('Temperature (C)')
        PLT.legend()
        PLT.show()

        ##Perform Prediction
        ssdoPred = SSDO.SSDataObject(predictFC, useChordal = False)
        ssdoPred.obtainData(ssdoPred.oidName, predictVars)

        dataPredict = NUM.zeros((ssdoPred.numObs, len(predictVars)))

        for varNum, var in enumerate(predictVars):
            dataPredict[:,varNum] = ssdoPred.fields[var].data

        y_hat_downscale = ridge.predict(dataPredict)

        #writeFC(predictFC, outputFC, y_hat_downscale)

        return
