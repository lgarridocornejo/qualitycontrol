import pandas as pd
import numpy as np

class duplicados:

    def __init__(self, df, orgid, org, dupid, dup):
        """
        """
        self.dfqc = df[[orgid,org,dupid,dup]]
        self.samporg = orgid
        self.org = org
        self.sampdup = dupid
        self.dup = dup

    def proceso(self):
        """
        """
        self.delta()
        self.maxmin()
        self.eralativ()
        self.promedio()
        return self.dfqc
        
    def maxmin(self):
        """
        """
        #var = self.analyte + '_max'
        self.dfqc['max'] = np.maximum(self.dfqc[self.org],self.dfqc[self.dup])
        #var = self.analyte + '_min'
        self.dfqc['min'] = np.minimum(self.dfqc[self.org],self.dfqc[self.dup])

    def eralativ(self):
        """
        """
        #var = self.analyte + '_erel'
        self.dfqc['erel'] = 2*abs(self.dfqc[self.org]-self.dfqc[self.dup])/(self.dfqc[self.org]+self.dfqc[self.dup])

    def promedio(self):
        """
        """
        #var = self.analyte + '_mean'
        self.dfqc['media'] = np.add(self.dfqc[self.org],self.dfqc[self.dup])/2

    def delta(self):
        """
        """
        #var = self.analyte + '_mean'
        self.dfqc['delta'] = np.abs(self.dfqc[self.org]-self.dfqc[self.dup])

    def tol_lineal(self, TOL):
        """
        """
        #self.dfqc['tol_lin'] = 100 * self.dfqc['min'] / (100-TOL)
        self.dfqc['lineal'] = self.dfqc[['min','max']].apply(lambda x: 'Fallo' if (x['max'] - (100*x['min']/(100-TOL)) > 0) else 'Muestra',axis=1)
        
    def tol_hiperb(self, m,b):
        """
        """
        #self.dfqc['tol_hip'] = 100 * self.dfqc['min'] / (100-TOL)
        self.dfqc['hiperb'] = self.dfqc[['min','max']].apply(lambda x: 'Fallo' if (x['max'] - (m * x['min']**2 + b)**0.5 > 0) else 'Muestra',axis=1)      
                                                                                        # (m2 * ordenada**2 + b2)**0.5

if __name__ == '__main__':
    main()
