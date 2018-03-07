import json
import pandas
import root_pandas as rp
import os
import sys
import yaml


def main():

    SR = Reader("tt","conf/scale_samples.json",2)
    with open("dump.json","w") as FSO:
        json.dump(SR.config, FSO, indent = 4, sort_keys=True )

class Reader():

    def __init__(self, channel,config_file, folds):
        self.itersamples = []
        self.idx = 0

        self.channel = channel
        self.trainReweighting = ""
        self.folds = folds
        self.processes = []
        
        with open("conf/hist_names.json","r") as FSO:
            self.hist_names = json.load(FSO)

        with open("conf/cuts.json","r") as FSO:
            cuts = json.load(FSO)
            for c in cuts:
                cuts[c] = self._assertChannel( cuts[c] )
            self.cut_dict = cuts

        self.config = self._flattenConfig(config_file)


    def __iter__(self):
        return self

    def next(self):
        try:
            sample = self.itersamples[ self.idx ]
            self.idx += 1
            return self.loadForMe( sample ), sample["histname"]
        except IndexError as e:
            raise StopIteration


    def _flattenConfig(self,config_file):
        '''
        Read dataset configuration file and flatten the return object to the current use case.
        '''
        try:
            with open(config_file,"r") as FSO:
                config = json.load(FSO)
        except ValueError as e:
            print e
            print "Check {0}. Probably a ',' ".format(config_file)
            sys.exit(0)

        targets = []
        config["channel"] = self.channel
        config["target_names"] = {}
        self.trainReweighting = config["train_weight"] 

        config["path"] = "{path}/ntuples_{version}/{channel}/ntuples_{useSV}_merged".format( **config )

        for sample in config["samples"]:
            
            self.processes.append(sample)

            sample_name = self._assertChannel( config["samples"][sample]["name"] )
            config["samples"][sample]["target"] = self._assertChannel(config["samples"][sample]["target"] )
            targets.append( config["samples"][sample]["target"]  )

            config["samples"][sample]["name"]    = "{path}/{name}_{channel}_{version}.root".format(name = sample_name, **config)
            config["samples"][sample]["select"] = self._parseCut( config["samples"][sample]["select"] )
            
            if sample != "data" and sample != "data_ss":
                config["samples"][sample]["shapes"]  = self._getShapePaths( config["samples"][sample]["name"] )
                if type(config["samples"][sample]["event_weight"]) is list:
                    config["addvar"] = list( set( config["addvar"] + config["samples"][sample]["event_weight"] ) )

        targets.sort()
        targets = [ t for t in targets if t != "none" ]
        target_map = {}

        for i,t in enumerate(set(targets)):
            config["target_names"][i] = t
            target_map[t] = i

        for sample in config["samples"]:
            config["samples"][sample]["target"]  = target_map.get( config["samples"][sample]["target"], "none" )  

        return config


    def getSamplesForTraining(self):
        self.setNominalSamples()
        samples = []
        for sample,histname in self:
            samples.append(sample)
        print "Combining for training"
        return self.combineFolds(samples)

    def setNominalSamples(self):
        self.itersamples = []
        self.idx = 0
        samples = self.config["samples"].keys()
        samples.sort()
        for sample in samples:
            if sample == "data" or "_more" in sample: continue

            tmp = self._getCommonSettings(sample)

            tmp["path"] = self.config["samples"][sample]["name"] 
            tmp["histname"   ] = sample
            tmp["rename"      ] = {}

            self.itersamples.append( tmp )

        return self

    def setLooserSamples(self):
        self.itersamples = []
        self.idx = 0
        samples = self.config["samples"].keys()
        samples.sort()
        for sample in samples:
            if not "_more" in sample: continue

            tmp = self._getCommonSettings(sample)

            tmp["path"] = self.config["samples"][sample]["name"] 
            tmp["histname"   ] = sample
            tmp["rename"      ] = {}

            self.itersamples.append( tmp )

        return self

    def setDataSample(self):
        self.itersamples = []
        self.idx = 0

        tmp = self._getCommonSettings("data")

        tmp["path"] = self.config["samples"]["data"]["name"] 
        tmp["histname"   ] = "data_obs"
        tmp["rename"      ] = {}

        self.itersamples.append( tmp )

        return self

    def setTESSamples(self):
        self.itersamples = []
        self.idx = 0
        samples = self.config["samples"].keys()
        samples.sort()
        for sample in samples:
            if sample == "data" or sample == "estimate" or "_more" in sample: continue

            for shape in self.config["samples"][sample]["shapes"]:
                if "JEC" in shape: continue

                tmp = self._getCommonSettings(sample)

                tmp["path"] = self.config["samples"][sample]["shapes"][shape] 
                tmp["histname"   ] = sample + self.hist_names[shape]
                tmp["rename"      ] = {}

                self.itersamples.append( tmp )

        return self

    def setJECSamples(self):
        self.itersamples = []
        self.idx = 0
        samples = self.config["samples"].keys()
        samples.sort()
        for sample in samples:
            if sample == "data" or sample == "estimate" or "_ss" in sample: continue

            for shape in self.config["samples"][sample]["shapes"]:
                if not "JEC" in shape: continue

                tmp = self._getCommonSettings(sample)

                tmp["path"] = self.config["samples"][sample]["shapes"][shape] 
                tmp["histname"   ] = sample + self.hist_names[shape]
                tmp["rename"      ] = self._getRenaming( shape.replace("JEC","") )

                self.itersamples.append( tmp )

        return self


    def loadForMe(self, sample_info):


        DF = self._getDF(sample_path = sample_info["path"], 
                          select = sample_info["select"])

        print "Loading ", sample_info["histname"], len(DF)

        DF.eval( "event_weight = " + sample_info["event_weight"], inplace = True  )

        # DF["histname"] = sample_info["histname"]
        DF["target"] = sample_info["target"]

        if not self.trainReweighting:
            DF["train_weight"] = 1.0
        else:
            DF["train_weight"] = self._getTrainWeight(DF, scale = sample_info["train_weight_scale"] )

        if sample_info["rename"]:
            DF.rename(columns = sample_info["rename"], inplace = True)

        return self._getFolds( DF )



    def combineFolds(self, samples):

        folds = [ [fold] for fold in samples[0] ]
        for sample in samples[1:]:
            for i in xrange(len(folds)):
                folds[i].append( sample[i] )

        for i,fold in enumerate(folds): 
            folds[i] = pandas.concat( fold, ignore_index=True).sample(frac=1., random_state = 41).reset_index(drop=True)

        return folds

    def get(self, what):
        if what == "nominal"  : return self.setNominalSamples()
        if what == "more"     : return self.setLooserSamples()
        if what == "data"     : return self.setDataSample()
        if what == "tes"      : return self.setTESSamples()
        if what == "jec"      : return self.setJECSamples()

    def _parseCut(self, cutstring):
        cutstring = self._assertChannel( cutstring )
        for alias,cut in self.cut_dict.items():
            cutstring = cutstring.replace( alias, cut )
        return cutstring

    def _assertChannel(self, entry):

        if type( entry ) is dict:
            return entry[ self.channel ]
        else:
            return entry      

    def _getShapePaths(self, sample):

        shapes = {"T0Up":"","T0Down":"","T1Up":"","T1Down":"","T10Up":"","T10Down":"","JECUp":sample,"JECDown":sample}

        for shape in shapes:
            shape_path = sample.replace(".root","_{0}.root".format(shape) )
            if os.path.exists( shape_path ):
                shapes[shape] = shape_path
            else:
                shapes[shape] = sample

        return shapes

    def _getCommonSettings(self, sample):

        settings = {}
        settings["event_weight"] = self._getEventWeight(sample)
        settings["target"      ] = self.config["samples"][sample]["target"] 
        settings["select"      ] = self.config["samples"][sample]["select"]
        settings["train_weight_scale"] = self.config["samples"][sample]["train_weight_scale"]
        
        return settings


    def _getEventWeight(self, sample):
        if type( self.config["samples"][sample]["event_weight"] ) is list:
            return "*".join( self.config["samples"][sample]["event_weight"] + [ str(self.config["lumi"]) ] )

        if type( self.config["samples"][sample]["event_weight"] ) is float:
            return str( self.config["samples"][sample]["event_weight"] )

        else:
            return 1.0

    def _getTrainWeight(self, DF, scale):
        if self.trainReweighting == "normalize_evt":
            evts = len(DF)
            if evts > 0: return 10000 / float(evts)

        elif self.trainReweighting == "normalize_xsec":
            return DF["event_weight"].abs() * scale

        elif self.trainReweighting == "use_scale":
            return scale

        else:
            return 1.0

    def _getRenaming(self, corr):

        tmp =[]
        for nom in ["mjj","jdeta","njets","jpt"]:
            if not nom == "jpt": tmp+= [(nom, nom+corr),(nom+corr, nom) ]
            else : tmp += [ (nom + "_1", nom+corr+ "_1"),(nom+corr+ "_1", nom+ "_1"),
                            (nom + "_2", nom+corr+ "_2"),(nom+corr+ "_2", nom+ "_2") ]

        return dict( tmp )

    def _getFolds(self, df):

        if self.folds != 2: raise NotImplementedError("Only implemented two folds so far!!!")
        folds = []

        folds.append( df.query( "fileEntry % 2 != 0 " ).reset_index(drop=True) )
        folds.append( df.query( "fileEntry % 2 == 0 " ).reset_index(drop=True) )

        return folds

    def _getDF( self, sample_path, select ):

        branches = set( self.config["variables"] + self.config["addvar"] )
        tmp = rp.read_root( paths = sample_path,
                             where = select,
                             columns = branches)

        tmp.replace(-999.,-10, inplace = True)
        return tmp

if __name__ == '__main__':
    main()