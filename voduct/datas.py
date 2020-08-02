from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from keras.utils.data_utils import get_file
import os
import copy
import pickle
import tokenizer as tk
import voduct.models as vomods



def rand_sample(arr, n_samples=1):
    """
    Randomly samples a single element from the argued array.

    arr: sequence of some sort
    """
    if not isinstance(arr,list): arr = list(arr)
    if len(arr) == 0: print("len 0:", arr)
    samples = []
    perm = np.random.permutation(len(arr))
    for i in range(n_samples):
        samples.append(arr[perm[i]])
    if len(samples) == 1: return samples[0]
    return samples

class Tokenizer():
    """
    This class assists in tokenizing the data and converting between
    indices and tokens.
    """
    def __init__(self, X=None, Y=None, word2idx=None,
                                       idx2word=None,
                                       split_digits=False,
                                       MASK="<MASK>",
                                       START="<START>",
                                       STOP="<STOP>",
                                       INIT="<INIT>",
                                       seq_len_x=None,
                                       seq_len_y=None,
                                       prepend=False,
                                       append=False,
                                       index=True,
                                       verbose=True):
        """
        X: list of strings
            these are the strings that will be tokenized and indexed.
            if None is argued, the word2idx and idx2word must not be
            None
        Y: list of strings
            these are another list of strings that will be tokenized
            and indexed. the words in Y will be be included in the 
            vocabulary if word2idx and idx2word are None
        word2idx: dict
            keys: str
                the words or tokens
            values: int
                the integer indices corresponding to each token
        idx2word: dict
            keys: int
                the integer indices corresponding to each token
            values: str
                the words or tokens corresponding to each index
        split_digits: bool
            option to split each digit into a sequence of individual
            digits 0-9
        MASK: str
            the token to correspond to the null embedding
        START: str
            the token to correspond to the START embedding
        STOP: str
            the token to correspond to the STOP embedding
        INIT: str
            a token that optionally all decoding inputs begin with
        seq_len_x: int or None
            if None, then the maximum length of the tokenized X
            will be used for the X sequence length
        seq_len_y: int or None
            if None, then the maximum length of the tokenized Y
            will be used for the Y sequence length
        prepend: bool
            if true, self.START is prepended to the start of the tokens
        append: bool
            if true, self.STOP is appended to the end of the tokens
        index: bool
            if true, the tokens are also converted to indices
        """
        self.MASK = MASK
        self.START = START
        self.STOP = STOP
        self.INIT = INIT
        self.string_X = X
        self.string_Y = Y
        if split_digits:
            words = set([str(i) for i in range(10)])
        else:
            words = set([str(i) for i in range(100)])

        if X is not None or Y is not None:
            x_max_len = 0
            y_max_len = 0
            tok_x = []
            tok_y = []
            n_loops = len(X) if X is not None else len(Y)
            if verbose:
                print("Tokenizing")
            for i in range(n_loops):
                tok_x.append(tk.tokenize(X[i],
                             split_digits=split_digits))
                words = words | set(tok_x[i])
                x_max_len = max(x_max_len,len(tok_x[i]))
                tok_y.append(tk.tokenize(Y[i],
                             split_digits=split_digits))
                words = words | set(tok_y[i])
                y_max_len = max(y_max_len,len(tok_y[i]))
                if verbose:
                    print(round(float(i)/len(X)*100),"%",
                                                    end="    \r")
            self.token_X = tok_x
            self.token_Y = tok_y
        if word2idx is None:
            word2idx = {w:i+1 for i,w in enumerate(words)}
            word2idx[self.MASK] = 0
        if idx2word is None:
            idx2word = {i+1:w for i,w in enumerate(words)}
            idx2word[0] = self.MASK
        if self.START not in word2idx:
            word2idx[self.START] = len(word2idx)
            idx2word[word2idx[self.START]] = self.START
        if self.STOP not in word2idx:
            word2idx[self.STOP] = len(word2idx)
            idx2word[word2idx[self.STOP]] = self.STOP
        if self.INIT not in word2idx:
            word2idx[self.INIT] = len(word2idx)
            idx2word[word2idx[self.INIT]] = self.INIT
        self.INIT_IDX = word2idx[self.INIT]
        self.word2idx = word2idx
        self.idx2word = idx2word
        if (X is not None or Y is not None) and index:
            if verbose:
                print("Converting to integer indexes")
            seq_len_x = x_max_len if seq_len_x is None else seq_len_x
            self.X = self.index_tokens(tok_x, seq_len_x+prepend+append,
                                              prepend=prepend,
                                              append=append)
            seq_len_y = y_max_len if seq_len_y is None else seq_len_y
            self.Y = self.index_tokens(tok_y, seq_len_y+prepend+append,
                                              prepend=prepend,
                                              append=append)
            self.inits = [self.INIT for i in range(seq_len_y+append)]
            self.inits = self.index_tokens([self.inits],
                                            seq_len_y+prepend+append,
                                            prepend=prepend,
                                            append=False)


    def index_tokens(self, toks, seq_len, prepend=False,
                                          append=False,
                                          verbose=True):
        """
        Used to convert tokens to indices

        toks: list of lists of str (N, variable)
            the tokens to be indexed
        seq_len: int
            the length of the sequence
        prepend: bool
            if true, self.START is prepended to the start of the tokens
        append: bool
            if true, self.STOP is appended to the end of the tokens
        """
        X = torch.zeros(len(toks),seq_len).long()
        for i in range(len(toks)):
            if prepend: X[i,0] = self.word2idx[self.START]
            for j,x in enumerate(toks[i]):
                X[i,j+prepend] = self.word2idx[x]
            if append:
                X[i,min(j+1+prepend,seq_len-1)]=self.word2idx[self.STOP]
            if verbose:
                print(round(float(i)/len(toks)*100),"%",
                                             end="    \r")
        return X

class WordProblems(Dataset):
    def __init__(self, difficulty="easy", lowercase=True,
                                          n_samples=10000,
                                          max_count=1000,
                                          split_digits=False,
                                          verbose=True,
                                          index=True,
                                          make_structs=False,
                                          **kwargs):
        """
        This is a class to assist in automatic generation of word
        problems. Each entry consists of a problem setup with a
        question. All entries and answers are strings.
        
        Examples:
        x = "you move {n1} {obj_type1} to the goal, then you move
        {n2} {obj_type2} to the goal, ... 
        How many objects are at the goal?"
        y = "{n1+n2+...} objects"

        difficulty: str
            the difficulty of the dataset
        lowercase: bool
            if true, all letters are lowercase
        n_samples: int
            the number of samples for the dataset
        max_count: int
            the maxium number for the word problems to use
        split_digits: bool
            split digits into individual 0-9 tokens
        index: bool
            if false, the sentence based questions and answers are
            not tokenized and converted to indices.
        make_structs: bool
            if true, returns a list of dictionaries of sample questions
            broken into their sentence parts with the corresponding
            count dicts for each action, and saved as a member variable
            self.samp_structs
        """
        self.tokenizer = Tokenizer()
        self.lowercase = lowercase
        self.n_samples = n_samples
        self.max_count = max_count
        rand_labels = False
        if difficulty == "rand" or difficulty == "random":
            rand_labels = True
            difficulty = "easy"
            

        self.attrs = ["color", "shape", "all", "none"]
        self.colors=["red","blue","green","yellow","orange","purple"]
        self.shapes = ["box", "ball", "cylinder", "coin", "ramp"]
        self.obj_types = []
        for color in self.colors:
            for shape in self.shapes:
                self.obj_types.append((color+" "+shape).strip())

        self.initial_conditions = [
            "<count> <type> objects start at <loc>",
            "<count> <type> objects are initialized at <loc>",
            #"you move <count> <type> objects to <loc>",
            #"you take <count> <type> objects to <loc>",
            #"you put <count> <type> objects at <loc>",
            ]

        self.action_statements = [
            "you <verb> <count> <type> objects <preposition1> "+\
                                 "<loc1> <preposition2> <loc2>"
            ]

        self.verbs =  ["move", "take"]
        self.qverbs = ["are at", "started at", "ended at"]
                  #"existed at some point at"]
        self.pos_prepositions = ["to"]
        self.neg_prepositions = ["from", "away from"]
        self.prepositions = [self.pos_prepositions,self.neg_prepositions]
        self.locations = ["the starting point","the goal"]

        self.directions = {
            "move":["in front of","left of","right of","on top of",
                     "behind"], 
            "remove": ["from"],
            "take":["from", "away from"]}

        if verbose:
            print("Making data")
        samples = self.make_data_set(n_samples=n_samples,
                                     lowercase=lowercase,
                                     difficulty=difficulty,
                                     max_count=max_count,
                                     make_structs=make_structs,
                                     verbose=verbose)
        self.samp_structs = samples[-1]
        samples = [list(zip(*s)) for s in samples[:-1]]
        start_samps, end_samps, move_samps = samples
        questions = start_samps[0]+end_samps[0]+move_samps[0]
        answers = start_samps[1]+end_samps[1]+move_samps[1]
        
        self.samples = {"start":start_samps, "end":end_samps,
                                             "move":move_samps}
        self.questions = questions
        self.answers = answers
        if index:
            self.tokenizer = Tokenizer(X=questions,
                                       Y=answers,
                                       split_digits=split_digits,
                                       prepend=True,
                                       append=True)
            self.X = self.tokenizer.X
            if rand_labels:
                Y = self.tokenizer.Y
                for i in range(len(Y)):
                    Y[i][1:3] = torch.randint(len(self.idx2word),(2,))
                self.tokenizer.Y = Y
            self.Y = self.tokenizer.Y
            self.questions = self.tokenizer.string_X
            self.answers = self.tokenizer.string_Y
            self.token_qs = self.tokenizer.token_X
            self.token_ans = self.tokenizer.token_Y

    @property
    def inits(self):
        return self.tokenizer.inits
    @property
    def word2idx(self):
        return self.tokenizer.word2idx
    @property
    def idx2word(self):
        return self.tokenizer.idx2word
    @property
    def MASK(self):
        return self.tokenizer.MASK
    @property
    def START(self):
        return self.tokenizer.START
    @property
    def STOP(self):
        return self.tokenizer.STOP
                        
    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        return self.X[idx],self.Y[idx]

    def get_init(self, count_dict, max_count=10,
                                   min_count=1,
                                   obj_types=None,
                                   locs=None,
                                   n_inits=1,
                                   allow_dups=True):
        """
        Creates an inital condition

        count_dict: dict
            keys: str
                moved, end, start
                these are the potential things to keep track of. start
                tracks the objects that started at different locations.
                end tracks object counts as they are moved around the
                different locations. moved tracks the total times an
                object type is moved.
            vals: dict
                keys: str
                    the locations or the object types for "moved"
                vals: dict
                    keys: str
                        the object types
                    vals: int
                        the counts of the type at that location
        max_count: int
            the maximum possible number of objects to be initialized.
        min_count: int
            the minimum possible number of objects to be initialized.
        obj_types: sequence or None
            this is an optional argument that can be used to restrict
            the setting to particular object types. This is useful if
            you want a particularly difficult problem.
        locs: sequence or None
            this is an optional argument that can be used to restrict
            the setting to particular locations.
        n_inits: int
            if you want to instantiate multiple objects in one go.
        allow_dups: bool
            if true, then objects of the same type can be initialized
            mulitple times at the same location.
        """
        if n_inits <= 0: n_inits = 1
        if max_count is None: max_count = 10
        if min_count is None: min_count = 1
        if locs is not None: loc = rand_sample(locs)
        else: loc = rand_sample(self.locations)
        condition = rand_sample(self.initial_conditions)
        if n_inits > 1:
            condition = condition.replace("<count> <type>",
                                          "<count1> <type1>")
        for i in range(n_inits):
            count = rand_sample(range(min_count, max_count+1))

            if obj_types is not None:
                obj_type = rand_sample(obj_types)
            else:
                obj_types = set(self.obj_types)
                if not allow_dups:
                    for obj in count_dict["start"][loc]["all"].keys():
                        if obj in obj_types:
                            obj_types.remove(obj)
                try:
                    obj_type = rand_sample(list(obj_types))
                except:
                    print("Duplicating obj types due to insufficient types")
                    obj_type = rand_sample(self.obj_types)

            color,shape = [o.strip() for o in obj_type.split(" ")]
            count_dict["start"][loc]["all"][obj_type] += count
            count_dict["start"][loc]["color"][color] += count
            count_dict["start"][loc]["shape"][shape] += count
            count_dict["start"][loc]["none"] += count
            count_dict["end"][loc]["all"][obj_type] += count
            count_dict["end"][loc]["color"][color] += count
            count_dict["end"][loc]["shape"][shape] += count
            count_dict["end"][loc]["none"] += count
            if "you move" in condition or "you take" in condition or\
                    "you put" in condition:
                count_dict["moved"]["all"][obj_type] += count
                count_dict["moved"]["color"][color] += count
                count_dict["moved"]["shape"][shape] += count
                count_dict["moved"]["none"] += count
                
            keys = ["<count>","<type>","<loc>"]
            vals = [count, obj_type, loc]
            info = {k:v for k,v in zip(keys,vals)}
            if i < n_inits-2:
                condition = condition.replace("<count1> <type1>",
                              "<count> <type>, <count1> <type1>")
                for k,v in info.items():
                    condition = condition.replace(k,str(v))
            elif i < n_inits-1:
                condition = condition.replace("<count1> <type1>",
                              "<count> <type>, and <count1> <type1>")
                for k,v in info.items():
                    condition = condition.replace(k,str(v))
            else:
                condition = condition.replace("<count1> <type1>",
                                              "<count> <type>")
                for k,v in info.items():
                    condition = condition.replace(k,str(v))

        if count == 1 and n_inits == 1:
            condition = condition.replace("objects", "object")
            condition = condition.replace("start ", "starts ")
            condition = condition.replace("are ", "is ")
        return condition, count_dict, info

    def get_action(self, count_dict, start_loc=None,
                                     end_loc=None,
                                     obj_types=None):
        """
        Creates an action statement

        count_dict: dict
            keys: str
                moved, end, start
                these are the potential things to keep track of. start
                tracks the objects that started at different locations.
                end tracks object counts as they are moved around the
                different locations. moved tracks the total times an
                object type is moved.
            vals: dict
                keys: str
                    the locations or the object types for "moved"
                vals: dict
                    keys: str
                        the object types
                    vals: int
                        the counts of the type at that location
        obj_types: sequence or None
            this is an optional argument that can be used to restrict
            the action to particular object types. This is useful if
            an initialization condition was used or if you want a
            particularly difficult problem.
        """
        if start_loc is None or end_loc is None:
            keys = count_dict["end"].keys()
            start_loc,end_loc = rand_sample(keys,2)
        objs = []
        objs_was_none = obj_types is None
        while len(objs) == 0:
            objs = []
            counts = count_dict["end"][start_loc]["all"]
            if objs_was_none:
                obj_types = set(counts.keys())
            for obj in counts.keys():
                if counts[obj] > 0 and obj in obj_types:
                    objs.append(obj)
            if len(objs) == 0:
                keys = count_dict["end"].keys()
                start_loc,end_loc = rand_sample(keys,2)
        assert start_loc != end_loc
        obj_type = rand_sample(objs)
        count = rand_sample(list(range(counts[obj_type]+1)))
        prep1 = rand_sample(self.neg_prepositions)
        prep2 = rand_sample(self.pos_prepositions)
        verb = rand_sample(self.verbs)
        color,shape = [o.strip() for o in obj_type.split(" ")]
        count_dict["end"][start_loc]["all"][obj_type] -= count
        count_dict["end"][start_loc]["color"][color] -= count
        count_dict["end"][start_loc]["shape"][shape] -= count
        count_dict["end"][start_loc]["none"] -= count
        count_dict["end"][end_loc]["all"][obj_type] += count
        count_dict["end"][end_loc]["color"][color] += count
        count_dict["end"][end_loc]["shape"][shape] += count
        count_dict["end"][end_loc]["none"] += count
        count_dict["moved"]["all"][obj_type] += count
        count_dict["moved"]["color"][color] += count
        count_dict["moved"]["shape"][shape] += count
        count_dict["moved"]["none"] += count
        info = {"type":obj_type,
                "count":count,
                "to":end_loc,
                "from":start_loc}
        # For variety
        if np.random.random() > .5:
            prep1,prep2 = prep2,prep1
            start_loc,end_loc = end_loc,start_loc

        statement = rand_sample(self.action_statements)
        if count == 1: statement=statement.replace("objects", "object")
        keys = ["<verb>", "<count>", "<type>","<preposition1>",
                "<loc1>", "<preposition2>", "<loc2>"]
        vals = [verb, str(count), obj_type, prep1, start_loc,
                                                   prep2, end_loc]
        for k,v in zip(keys,vals):
            statement = statement.replace(k, v)
        return statement, count_dict, info

    def get_count_dict(self):
        """
        Returns a dictionary that is compatible with the make_data_set
        func. The returned dict has the following structure:

        {
            "start":{
                "loc1":{
                    "color":{
                        "red":0,
                        "blue":...
                    },
                    "shape":{
                        "cylinder":0,
                        "box":...
                    },
                    "all":{
                        "red cylinder":0,
                        "red box":...
                    }
                    "none":0 # sum of all objects
                },
                "loc2":...
            },
            "end":{
                "loc1":{
                    "color":{
                        "red":0,
                        "blue":...
                    },
                    "shape":{
                        "cylinder":0,
                        "box":...
                    },
                    "all":{
                        "red cylinder":0,
                        "red box":...
                    }
                    "none":0 # sum of all objects
                },
                "loc2":...
            }
            "moved":{
                "color":{
                    "red":0,
                    "blue":...
                },
                "shape":{
                    "cylinder":0,
                    "box":...
                },
                "all":{
                    "red cylinder":0,
                    "red box":...
                }
                "none":0 # sum of all objects
            }
        """
        def zero(): return 0
        attrs = self.attrs
        moved_dict = {attr:defaultdict(zero) for attr in attrs}

        for obj in self.obj_types:
            color, shape = obj.split(" ")
            moved_dict["color"][color.strip()] = 0
            moved_dict["shape"][shape.strip()] = 0
            moved_dict["all"][obj] = 0
            moved_dict["none"] = 0
        count_dict ={
            "start":{l:{attr:defaultdict(zero) for attr in attrs}\
                                         for l in self.locations},
            "end":  {l:{attr:defaultdict(zero) for attr in attrs}\
                                         for l in self.locations},
            "moved":moved_dict
            }
        for loc in self.locations:
            count_dict["start"][loc]["none"] = 0
            count_dict["end"][loc]["none"] = 0
        return count_dict

    def make_data_set(self, n_samples=5000, lowercase=True,
                                            difficulty="easy",
                                            max_count=1000,
                                            make_structs=False,
                                            verbose=True):
        """
        Makes an easy dataset. Numbers and answers are always 5 or
        less. The form of the statements is always 1 sub-statement
        maximum. The locations are either the goal or the starting
        point.

        n_samples: int
            the number of training examples
        lowercase: bool
            if true, all letters are lowercase
        difficulty: str
            easy, medium or hard
        make_structs: bool
            if true, returns a list of dictionaries of sample questions
            broken into their sentence parts with the corresponding
            count dicts for each action

        Returns:
            <generic>_samples: set of (q,a) tuples
                each sample set contains string question answer tuples
            samp_structs: list of dicts
                each dict has two keys:
                    "init": dict()
                        "string":the actual initialization string
                        "counts":corresponding count dict
                    "actions": dict()
                        "strings": list of the action strings
                        "counts": list of the corresponding count dict
                            of each action and all actions leading up to
                            this point.
                        "infos": list of the corresponding info dict
                            of each action        
        """
        samp_structs = []
        start_samples = set()
        end_samples = set()
        move_samples = set()
        samples = [start_samples, end_samples, move_samples]
        samp_len = np.sum([len(s) for s in samples])
        n_loops = 0
        while samp_len < n_samples or n_loops>(n_samples+100000):
            n_loops += 1
            if make_structs:
                samp_struct = {"init":{"string":None, "counts":None},
                               "actions":{ "strings": [],
                                           "counts": [],
                                           "infos":  [] } }
            count_dict = self.get_count_dict()

            # Initial conditions
            sentence = ""
            if difficulty.lower() == "easy":
                n_inits = np.random.randint(1,3)
                allow_dups = False
            elif difficulty.lower() == "medium":
                n_inits = np.random.randint(2,5)
                allow_dups = True
            else:
                n_inits = np.random.randint(1,10)
                allow_dups = True
            start_idx,end_idx = 0,1
            if np.random.random() > .5:
                start_idx,end_idx = 1,0

            # Initial Condition Loop
            while n_inits > 0:
                rand_amt = np.random.randint(1,n_inits+1)
                init_condish,count_dict,info= self.get_init(count_dict,
                                                  max_count=max_count,
                                                  min_count=1,
                                                  locs=self.locations,
                                                  n_inits=rand_amt,
                                                  allow_dups=allow_dups)
                n_inits -= rand_amt
                sentence += init_condish
                if n_inits > 0: sentence += " and "
            sentence += ". "
            if make_structs:
                samp_struct['init']["strings"] = sentence
                temp_dict = copy.deepcopy(count_dict)
                del temp_dict["start"]
                samp_struct['init']["counts"] = temp_dict

            if difficulty.lower() == "easy":
                n_inits = np.random.randint(1,3)
            elif difficulty.lower() == "medium":
                n_inits = np.random.randint(2,6)
            else:
                n_inits = np.random.randint(5,10)

            # Action Statement Loop
            for i in range(n_inits):
                tup = self.get_action(count_dict=count_dict,
                                      start_loc=None,
                                      end_loc=None)
                action,count_dict,info = tup
                action = action + ". "
                sentence += action
                if i < n_inits-1: sentence += "then "
                if make_structs:
                    samp_struct['actions']["strings"].append(action)
                    temp_dict = copy.deepcopy(count_dict)
                    del temp_dict["start"]
                    samp_struct['actions']["counts"].append(temp_dict)
                    samp_struct['actions']["infos"].append(info)

            if make_structs:
                samp_structs.append(samp_struct)
            #starting_samples
            for loc in count_dict["start"].keys():
                s = "how many <type> objects started at {}?"
                question = s.format(loc)
                for attr in self.attrs:
                    if attr == "none":
                        q = question.replace("<type> ","")
                        q = sentence + q
                        count = count_dict["start"][loc][attr]
                        ans = "{} objects".format(count)
                        if count == 1: ans = ans[:-1]
                        if lowercase: 
                            q = q.lower()
                            ans = ans.lower()
                        start_samples.add((q,ans))
                        continue
                    for obj in count_dict["start"][loc][attr].keys():
                        q = question.replace("<type>",obj)
                        q = sentence + q
                        count = count_dict["start"][loc][attr][obj]
                        ans = "{} {} objects".format(count,obj)
                        if count == 1: ans = ans[:-1]
                        if lowercase: 
                            q = q.lower()
                            ans = ans.lower()
                        start_samples.add((q,ans))
            #ending samples
            for loc in self.locations:
                for attr in self.attrs:
                    s = "how many <type> objects are at {}?".format(loc)
                    if attr == "none":
                        q = s.replace("<type> ", "")
                        q = sentence + q
                        count = count_dict["end"][loc][attr]
                        ans = "{} objects".format(count)
                        if count == 1: ans = ans[:-1]
                        if lowercase: 
                            q = q.lower()
                            ans = ans.lower()
                        end_samples.add((q,str(ans)))
                        continue
                    for obj in count_dict["end"][loc][attr].keys():
                        q = s.replace("<type>",obj)
                        q = sentence + q
                        count = count_dict["end"][loc][attr][obj]
                        ans = "{} {} objects".format(count,obj)
                        if count == 1: ans = ans[:-1]
                        if lowercase: 
                            q = q.lower()
                            ans = ans.lower()
                        end_samples.add((q,str(ans)))
            #moving samples
            s = "how many <type> objects did you move?"
            for attr in self.attrs:
                if attr == "none":
                    q = s.replace("<type> ","")
                    q = sentence + q
                    count = count_dict["moved"][attr]
                    ans = "{} objects".format(count)
                    if count == 1: ans = ans[:-1]
                    if lowercase: 
                        q = q.lower()
                        ans = ans.lower()
                    move_samples.add((q,ans))
                    continue
                objs = set()
                for loc in count_dict["start"].keys():
                    objs |= set(count_dict["start"][loc][attr].keys())
                for obj in objs:
                    q = s.replace("<type>",obj)
                    q = sentence + q
                    count = count_dict["moved"][attr][obj]
                    ans = "{} {} objects".format(count,obj)
                    if count == 1: ans = ans[:-1]
                    if lowercase: 
                        q = q.lower()
                        ans = ans.lower()
                    move_samples.add((q,ans))
            samp_len = np.sum([len(s) for s in samples])
            if verbose:
                print(round(samp_len/float(n_samples)*100), "%",
                                                      end="     \r")
        return start_samples, end_samples, move_samples, samp_structs


class Journal(Dataset):
    def __init__(self, seq_len=10):
        file_name = "dataset.p"
        self.MASK = "<MASK>"
        tup = self.get_data(file_name=file_name, seq_len=seq_len)
        X,Y,word2idx,idx2word = tup
        self.X = X # (N, SeqLen)
        self.Y = Y # (N, SeqLen)
        self.word2idx = word2idx
        self.idx2word = idx2word

    def get_data(self, file_name, seq_len, **kwargs):
        """
        Creates a dataset that consists of an X and and Y with shape
        (N,S) where S is the sequence length. This allows for easy
        batching. Y has the same shape as X but is shifted forward by
        one word.
        """
        # Get and prepare data
        with open(file_name,'rb') as f:
            self.data_dict = pickle.load(f)
        temp_fxn = lambda k,v: "<STARTENTRY>\n{}:\n{}".format(k,v)
        self.text = [temp_fxn(k,v) for k,v in self.data_dict.items()]
        self.text = "<ENDENTRY>\n".join(self.text)

        tokens = tk.tokenize(self.text)
        words = set(tokens)
        print("Num unique words:", len(words))

        word2idx = {w:i+1 for i,w in enumerate(words)}
        word2idx[self.MASK] = 0
        idx2word = {i+1:w for i,w in enumerate(words)}
        idx2word[0] = self.MASK

        X = [[word2idx[tokens[i+j]] for j in range(seq_len)]\
                            for i in range(len(tokens)-seq_len-1)]
        Y = [[word2idx[tokens[i+j]] for j in range(seq_len)]\
                            for i in range(1,len(tokens)-seq_len)]
        X = torch.LongTensor(X)
        Y = torch.LongTensor(Y)

        assert len(X) == len(Y)
        return X, Y, word2idx, idx2word
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class SentenceJournal(Dataset):
    """
    Converts each sentence into it's own sequence. The Y matrix is
    the same as the X matrix except that each token is shifted forward
    by one space.
    """
    def __init__(self, seq_len=None):
        file_name = "dataset.p"
        self.MASK = "<MASK>"
        tup = self.get_data(file_name=file_name, seq_len=seq_len)
        X,Y,word2idx,idx2word = tup
        self.X = X # (N, SeqLen)
        self.Y = Y # (N, SeqLen)
        self.word2idx = word2idx
        self.idx2word = idx2word

    def get_data(self, file_name, seq_len=None, **kwargs):
        """
        Creates a dataset that consists of an X and and Y with shape
        (N,S) where S is the sequence length. This allows for easy
        batching. Y has the same shape as X but is shifted forward by
        one word.
        """
        # Get and prepare data
        with open(file_name,'rb') as f:
            self.data_dict = pickle.load(f)
        temp_fxn = lambda k,v: "<STARTENTRY>\n{}:\n{}".format(k,v)
        self.text = [temp_fxn(k,v) for k,v in self.data_dict.items()]
        self.text = "\n".join(self.text)

        sentences = tk.get_sent_arr(self.text) #list of token lists
        words = set()
        max_sent_len = 0
        n_long_sents = 0
        if seq_len is None: seq_len = np.inf
        for sent in sentences:
            words = words|set(sent)
            max_sent_len = max(max_sent_len,len(sent))
            n_long_sents += int(len(sent) > seq_len)
        print("Num unique words:", len(words))
        print("Max Sentence Len:", max_sent_len)
        seq_len = max_sent_len if seq_len == np.inf else seq_len
        X = np.zeros((len(sentences)+n_long_sents, seq_len))
        Y = np.zero_like(X)

        word2idx = {w:i+1 for i,w in enumerate(words)}
        word2idx[self.MASK] = 0
        idx2word = {i+1:w for i,w in enumerate(words)}
        idx2word[0] = self.MASK

        i = -1
        for sent in sentences:
            i += 1
            idxs = np.asarray([word2idx[w] for w in sent])
            if len(idxs) <= seq_len:
                X[i,:len(idxs)] = idxs
                Y[i,:len(idxs)-1] = idxs[1:]
            # TODO make compatible with idxs that are > n*seq_len
            else:
                X[i,:seq_len-1] = idxs[:seq_len-1]
                X[i,seq_len-1] = word2idx["<STOP>"]
                X[i+1,0] = word2idx["<START>"]
                diff = len(idxs)-seq_len-1
                X[i+1,1:diff] = idxs[seq_len-1:diff]

        X = [[word2idx[tokens[i+j]] for j in range(seq_len)]\
                            for i in range(len(tokens)-seq_len-1)]
        Y = [[word2idx[tokens[i+j]] for j in range(seq_len)]\
                            for i in range(1,len(tokens)-seq_len)]
        X = torch.LongTensor(X)
        Y = torch.LongTensor(Y)

        assert len(X) == len(Y)
        return X, Y, word2idx, idx2word
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class EmptyDataset(Dataset):
    def __init__(self, X, Y,**kwargs):
        self.X = X
        self.Y = Y
        for k,v in kwargs.items():
            setattr(self,k,v)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_data(seq_len=10, shuffle_split=False,
                         train_p=0.8,
                         dataset="Journal",
                         model_type=vomods.SEQ2SEQ,
                         n_samples=50000,
                         difficulty="medium",
                         split_digits=False,
                         max_count=1000,
                         **kwargs):
    """
    Returns two torch Datasets, one validation and one training.

    seq_len: int
        the length of word sequences
    dataset: str
        the name of the dataset to be used
    shuffle_split: bool
        if true, shuffles before the split
    train_p: float (0,1)
        the portion of data reserved for training
    model_type: str
        the type of model (autoencoder or seq2seq)
    n_samples: int
        some datasets allow a specifications of the quantity of data
        points
    split_digits: bool
        split digits into individual 0-9 tokens
    """

    dataset = globals()[dataset](seq_len=seq_len,
                                 difficulty=difficulty,
                                 n_samples=n_samples,
                                 split_digits=split_digits,
                                 max_count=max_count)
    if shuffle_split:
        perm = torch.randperm(len(dataset)).long()
    else:
        if dataset=="WordProblems":
            print("WARNING!!!! shuffle_split is off but is highly recommended for WordProblems datasets. PLEASE STOP THIS TRAINING SESSION AND TURN SHUFFLE SPLIT ON")
        perm = torch.arange(len(dataset)).long()
    split_idx = int(len(perm)*train_p)
    if len(perm)-split_idx > 30000: split_idx = len(perm)-30000
    train_idxs = perm[:split_idx]
    val_idxs = perm[split_idx:]

    word2idx,idx2word = dataset.word2idx, dataset.idx2word
    val_dataset = EmptyDataset(X=dataset.X, Y=dataset.Y,
                               word2idx=word2idx,
                               idx2word=idx2word)
    val_dataset.X = val_dataset.X[val_idxs]
    val_dataset.Y = val_dataset.Y[val_idxs]

    dataset = EmptyDataset(X=dataset.X, Y=dataset.Y,
                               word2idx=word2idx,
                               idx2word=idx2word,
                               inits=dataset.inits)
    dataset.X = dataset.X[train_idxs]
    dataset.Y = dataset.Y[train_idxs]
    return dataset, val_dataset








