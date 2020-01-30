import json
import pandas
import root_pandas as rp
import os
import sys
import yaml
from helper import calc

def main():

    SR = Reader("mt","conf/global_config_2016.json",2)
    with open("dump_mt.json","w") as FSO:
        json.dump(SR.config, FSO, indent = 4, sort_keys=True )

class Reader():

    def __init__(self, channel,config_file, folds, era = ''):
        self.itersamples = []
        self.idx = 0
        self.era = era
        self.channel = channel
        self.trainReweighting = ""
        self.folds = folds
        self.processes = []
        self.needToAddVars = []

        with open("conf/cuts_{0}.json".format(era),"r") as FSO:
            cuts = json.load(FSO)
            for c in cuts:
                cuts[c] = self._assertChannel( cuts[c] )
            self.cut_dict = cuts
        self.config_file = config_file
        self.config = self._flattenConfig()


    def __iter__(self):
        return self

    def next(self):
        try:
            sample = self.itersamples[ self.idx ]
            self.idx += 1
            return self.loadForMe( sample ), sample
        except IndexError as e:
            raise StopIteration


    def _flattenConfig(self):
        '''
        Read dataset configuration file and flatten the return object to the current use case.
        '''
        try:
            with open(self.config_file,"r") as FSO:
                config = json.load(FSO)
        except ValueError as e:
            print e
            print "Check {0}. Probably a ',' ".format(self.config_file)
            sys.exit(0)

        targets = []
        config["channel"] = self.channel
        config["path"] = config["path"].format(**config) # Can be dropped when configs are split per channel
        config["outpath"] = config["outpath"].format(**config)
        config["target_names"] = {}
        config["variables"] = self._assertChannel( config["variables"] )
        config["version"] = self._assertChannel( config["version"] )
        for cw in config["class_weight"]:
            config["class_weight"][cw] = self._assertChannel( config["class_weight"][cw] )


        for sample in config["samples"]:
            
            snap = config["samples"][sample]
            self.processes.append(sample)

            sample_name = self._assertChannel( snap["name"] )
            snap["target"] = self._assertChannel(snap["target"] )
            targets.append( snap["target"]  )
            
            snap["name"]    = "{path}/{channel}-{name}.root".format(name = sample_name, **config)

            snap["select"] = self._parseCut( snap["select"] )

            snap["event_weight"]  = self._assertChannel( snap["event_weight"] )
            
            if sample != "data" and sample != "estimate" and "_full" in sample:
                snap["shapes"]  = self._getShapePaths( snap["name"], sample, 
                                                       config["shape_from_file"], 
                                                       config["shape_from_tree"] )
                if type(snap["event_weight"]) is list:
                    config["addvar"] = list( set( config["addvar"] + snap["event_weight"] ) )

            config["samples"][sample] = snap

        targets.sort()
        targets = [ t for t in targets if t != "none" ]
        target_map = {"none":-1}

        for i,t in enumerate(set(targets)):
            config["target_names"][i] = t
            target_map[t] = i

        for sample in config["samples"]:
            config["samples"][sample]["target_name"] = config["samples"][sample]["target"]

            if "target_values" in config:
                config["samples"][sample]["target"]  = int(config["target_values"].get( config["samples"][sample]["target"], -1 ))
            else:
                config["samples"][sample]["target"]  = target_map.get( config["samples"][sample]["target"], -1 )  

            config["target_names"][ config["samples"][sample]["target"] ] = config["samples"][sample]["target_name"]

        return config


    def getSamplesForTraining(self):
        self.for_prediction = False
        self.setNominalSamples()
        samples = []
        for sample,histname in self:
            samples.append(sample)
        print "Combining for training"
        return self.combineFolds(samples)

    def setNominalSamples(self):
        self.addvar = []
        self.itersamples = []
        self.idx = 0
        samples = self.config["samples"].keys()
        samples.sort()
        for sample in samples:
            if sample == "data" or "_full" in sample: continue

            tmp = self._getCommonSettings(sample)

            tmp["path"] = self.config["samples"][sample]["name"] 
            tmp["histname"   ] = sample
            tmp["rename"      ] = {}
            tmp["tree_name"   ] = self.config["tree_name_prefix"]
            
            self.itersamples.append( tmp )

        return self

    def setFullSamples(self, add_jec = False):
        self.addvar = self.config["addvar"]

        # If shifts are needed add them in the tree
        for v in self.config["shifted_variables"]:
            # if v in self.config["shifted_variables"]:
                self.addvar.append(v+"*")

        self.itersamples = []
        self.idx = 0
        samples = self.config["samples"].keys()
        samples.sort()
        for sample in samples:
            if "_full" in sample:

                tmp = self._getCommonSettings(sample)
                tmp["path"] = self.config["samples"][sample]["name"] 
                tmp["histname"   ] = sample
                tmp["tree_name"   ] = self.config["tree_name_prefix"]
                tmp["rename"      ] = {}
                self.itersamples.append( tmp )

                if add_jec and not "data" in sample:
                    for shape in self.config["samples"][sample]["shapes"]:

                        shapename = shape.replace("Up","").replace("Down","")
                        if not shapename in self.config["shape_from_tree"]: continue
                        if "EMB" in sample and ("em" not in self.channel or ("em" in self.channel and not "escale" in shapename)) :  continue
                        tmp = self._getCommonSettings(sample)

                        tmp["tree_name"] = self.config["tree_name_prefix"] + "_" + shape
                        tmp["path"] = self.config["samples"][sample]["shapes"][shape] 
                        tmp["histname"   ] = sample
                        tmp["rename"      ] = self._getRenaming( shape )

                        self.itersamples.append( tmp )                

        return self

    def setTESSamples(self):
        self.addvar = self.config["addvar"]
        self.itersamples = []
        self.idx = 0
        samples = self.config["samples"].keys()
        samples.sort()
        for sample in samples:
            if "data" in sample or not "_full" in sample: continue

            for shape in self.config["samples"][sample]["shapes"]:
                shapename = shape.replace("Up","").replace("Down","")
                if not shapename in self.config["shape_from_file"]: continue

                tmp = self._getCommonSettings(sample)

                tmp["path"] = self.config["samples"][sample]["shapes"][shape]
                if not tmp["path"]: continue

                tmp["histname"   ] =  shape +"_ntuple_" + sample.replace("_full","")
                tmp["rename"      ] = {}

                self.itersamples.append( tmp )

        return self

    def loadForMe(self, sample_info):
        if not os.path.exists( sample_info["path"] ):
            print "\033[1;31mWarning:\033[0m ", constStrLen( sample_info["histname"] ) , sample_info["path"]
            return []
            
        print "\033[1;32mLoading:\033[0m ", constStrLen( sample_info["histname"] ) , sample_info["path"].split("/")[-1]
        DF = self._getDF(sample_path = sample_info["path"], 
                          select = sample_info["select"],
                          tree_name = sample_info["tree_name"])

        # A bit hacky. Return  iterator when predicting to reduce memory consumption
        # Otherweise split in folds

        if self.for_prediction:
            return DF

        self.modifyDF(DF, sample_info)

        return self._getFolds( DF[ self.config["variables"] + ["target","train_weight","evt","event_weight"] ] )

    def modifyDF(self, DF, sample_info):

        DF["evt"] = DF["evt"].astype('int64')
        DF.eval( "event_weight = " + sample_info["event_weight"], inplace = True  )
        DF["target"] = sample_info["target"]
        DF["train_weight"] = DF["event_weight"].abs() * self.config["class_weight"].get(sample_info["target_name"], 1.0 )
        DF.replace(-999.,-10, inplace = True)

        for new, old in sample_info["rename"]:
            if new in DF.columns.values.tolist() and old in DF.columns.values.tolist():
                DF[old] = DF[new]
            # else:
            #     print "cant rename {0} to {1}".format(old, new)

        if self.era == "2016":
            DF.replace({"jdeta":-10.},-1., inplace = True)
            DF.replace({"mjj":-10.},-11., inplace = True)
            DF.replace({"dijetpt":-10.},-11., inplace = True)

            DF.eval("jdeta =   (njets < 2) *-1  + (njets > 1 )*jdeta ", inplace=True)
            DF.eval("mjj =     (njets < 2) *-11 + (njets > 1 )*mjj ", inplace=True)
            DF.eval("dijetpt = (njets < 2) *-11 + (njets > 1 )*dijetpt ", inplace=True)
            DF.eval("jpt_1 =   (njets == 0)*-10 + (njets > 0 )*jpt_1 ", inplace=True)
            DF.eval("jpt_2 =   (njets < 2 )*-10 + (njets > 1 )*jpt_2 ", inplace=True)


        if self.era == "2017":
            DF.replace({"jdeta":-1.},-10., inplace = True)
            DF.replace({"mjj":-11.},-10., inplace = True)
            DF.replace({"dijetpt":-11.},-10., inplace = True)

            DF.eval("jdeta =   (njets < 2) *-10 + (njets > 1 )*jdeta ", inplace=True)
            DF.eval("mjj =     (njets < 2) *-10 + (njets > 1 )*mjj ", inplace=True)
            DF.eval("dijetpt = (njets < 2) *-10 + (njets > 1 )*dijetpt ", inplace=True)
            DF.eval("jpt_1 =   (njets == 0)*-10 + (njets > 0 )*jpt_1 ", inplace=True)
            DF.eval("jpt_2 =   (njets < 2 )*-10 + (njets > 1 )*jpt_2 ", inplace=True)            

        if self.era == "2018":
            DF.replace({"jdeta":-1.},-10., inplace = True)
            DF.replace({"mjj":-11.},-10., inplace = True)
            DF.replace({"dijetpt":-11.},-10., inplace = True)

            DF.eval("jdeta =   (njets < 2) *-10 + (njets > 1 )*jdeta ", inplace=True)
            DF.eval("mjj =     (njets < 2) *-10 + (njets > 1 )*mjj ", inplace=True)
            DF.eval("dijetpt = (njets < 2) *-10 + (njets > 1 )*dijetpt ", inplace=True)
            DF.eval("jpt_1 =   (njets == 0)*-10 + (njets > 0 )*jpt_1 ", inplace=True)
            DF.eval("jpt_2 =   (njets < 2 )*-10 + (njets > 1 )*jpt_2 ", inplace=True)            


    def combineFolds(self, samples):

        folds = [ [fold] for fold in samples[0] ]
        for sample in samples[1:]:
            for i in xrange(len(folds)):
                folds[i].append( sample[i] )

        for i,fold in enumerate(folds): 
            folds[i] = pandas.concat( fold, ignore_index=True).sample(frac=1., random_state = 41).reset_index(drop=True)

        return folds

    def get(self, what, add_jec = False, for_prediction = False):
        self.for_prediction = for_prediction
        if what == "nominal"  : return self.setNominalSamples()
        if what == "full"     : return self.setFullSamples(add_jec)
        if what == "tes"      : return self.setTESSamples()

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

    def _getShapePaths(self, path, sample, from_file, from_tree):

        shapes = {}

        for shift in ["Up","Down"]:
            for shape in from_file:
                shapes[shape+shift] = path.replace("NOMINAL",shape+shift )

            for shape in from_tree:
                shapes[shape+shift] = path

        return shapes

    def _getCommonSettings(self, sample):

        settings = {}
        settings["event_weight"] = self._getEventWeight(sample)
        settings["target"      ] = self.config["samples"][sample]["target"]
        settings["target_name" ] = self.config["samples"][sample]["target_name"] 
        settings["select"      ] = self.config["samples"][sample]["select"]
        
        return settings


    def _getEventWeight(self, sample):
        if type( self.config["samples"][sample]["event_weight"] ) is list:
            return "*".join( self.config["samples"][sample]["event_weight"] + [ str(self.config["lumi"]) ] )

        if type( self.config["samples"][sample]["event_weight"] ) is float:
            return str( self.config["samples"][sample]["event_weight"] )

        if type( self.config["samples"][sample]["event_weight"] ) is unicode:
            return "*".join([str( self.config["samples"][sample]["event_weight"] ), str(self.config["lumi"]) ])

        else:
            return 1.0

    def _getRenaming(self, corr):

        tmp =[]
        for nom in self.config["shifted_variables"]:
            tmp.append( (nom+corr, nom) )
        return tmp 

    def _getFolds(self, df):

        if self.folds != 2: raise NotImplementedError("Only implemented two folds so far!!!")
        return [df.query( "abs(evt % 2) != 0 " ).reset_index(drop=True), df.query( "abs(evt % 2) == 0 " ).reset_index(drop=True) ]

    def _getDF( self, sample_path, select, tree_name):

        add = "addvar"
        if "Embedd" in sample_path:
            add = "addvar_Embedding"

        snowflakes = ["evt"]
#        if "ggH" in sample_path:
#            snowflakes.append("THU*")
#            snowflakes.append("*NNLO*")

        branches = list(set( self.config["variables"] + self.config[ "weights" ] + snowflakes + self.addvar ))
        if "EMB" in sample_path and "sf*" in branches:
            branches.remove("sf*")
        
        # Return iterator when predicting samples
        chunksize = None
        if self.for_prediction:
            chunksize = 100000

        try:
            tmp = rp.read_root( paths = sample_path,
                                key = tree_name,
                                where = select,
                                columns = branches,
                                chunksize = chunksize)
        except OSError as e:
            print('\n', e, '\nSkipping this tree\n') 
            return

        # tmp.replace(-999.,-10, inplace = True)
        # tmp["evt"] = tmp["evt"].astype('int64')

        return tmp

def constStrLen(string):

    return string + " "*(40 - len(string) )


if __name__ == '__main__':
    main()
