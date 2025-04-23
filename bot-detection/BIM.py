from alignment.sequence import Sequence
from alignment.vocabulary import Vocabulary
from alignment.sequencealigner import SimpleScoring, GlobalSequenceAligner
import ast
import json
import gzip
import time, os
from tqdm import tqdm

# Adding Timeout
import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
        

if os.path.exists("./bim.json"):
    with open("./bim.json", 'r') as file:
        bim_data = json.load(file)
else:
    with open("./data/user.json", 'r') as file:
        user_data = json.load(file)

    # Prepapring Author - Commit Message Map for Generating Data for BIM

    bim_data = {}

    for username in user_data:
        bim_data[username] = []
        for repo in user_data[username]["commits"]:
            for commit in user_data[username]["commits"][repo]:
                bim_data[username].append(user_data[username]["commits"][repo][commit]['message'])

    with open('./bim.json', 'w') as f:
        json.dump(bim_data, f)
                    

bin_threshold = 40
id_threshold = 0.5      # 50 percent
max_bot_bin = 500

result = []

for key in tqdm(bim_data, total=len(bim_data), ncols=80, leave=True):
    author, msgs = key, bim_data[key]
    tqdm.write(f'{author}, {len(msgs)}')
    
    if len (msgs) <= 1:
        ost = ';'.join([author, str(len(msgs)), str(1), str(1)])+'\n'
        result.append(ost)
        continue
    elif len (msgs) > 1000:
        ost = ';'.join([author, str(len(msgs)), str(1), str(0)])+'\n'
        result.append(ost)
        continue
    
    bins = {}       
    bratio = 0
    i = 0
    try:
        with timeout(seconds=10):                
            for msg in msgs:
                i += 1
                if len(bins) == 0:
                    bins[0] = [(msg, 100)]
                elif len(msg) >= 50:
                    bins[len(bins)] = [(msg, 100)]
                    continue
                else: 
                    '''
                    # Create sequences to be aligned.
                    b = Sequence('what a beautiful day'.split())
                    a = Sequence('what a disappointingly bad day'.split())
                    '''
                    a = Sequence(msg.split())
                    added = False
                    brflag = False
                    for key in bins:
                        b = Sequence(bins[key][0][0].split()) #first eleman of the tuple in the list
                        # Create a vocabulary and encode the sequences.
                        v = Vocabulary()   
                        try:
                            aEncoded = v.encodeSequence(a)
                            bEncoded = v.encodeSequence(b)

                            # Create a scoring and align the sequences using global aligner.
                            scoring = SimpleScoring(2, -1)
                            aligner = GlobalSequenceAligner(scoring, -2)                    
                            score, encodeds = aligner.align(aEncoded, bEncoded, backtrace=True)                        

                            # Iterate over optimal alignments and print them.
                            pi_max = 0
                            score_ = 0
                            for encoded in encodeds:                               
                                alignment = v.decodeSequenceAlignment(encoded)
                                score_ = alignment.score
                                percentIdentity =  alignment.percentIdentity()
                                if percentIdentity > pi_max : pi_max = percentIdentity                    

                            if pi_max > bin_threshold:
        #                         print (pi_max)
                                bins[key].append((msg, percentIdentity)) # add b and similarity
                                added = True
                                break
                        except KeyboardInterrupt:
                            print ('KeyboardInterrupt')
                            break
                        except:
                            brflag = True
                            break
                    if brflag:
                        bins = {}
                        break
                    if added == False:
                        bins[len(bins)] = [(msg, 100)]
                    if len(bins) > max_bot_bin:                            
                        bratio = 1
                        break
                        
    except TimeoutError:
        print ('Timeout')
        ost = ';'.join([author, str(i), str(len(bins.keys())), str(bratio)])+'\n'
        result.append(ost)
        continue
            
    num_commits = len(msgs)
    ratio = max(len(bins.keys()) / num_commits, bratio)
    ost = ';'.join([author, str(num_commits), str(len(bins.keys())), str(ratio)])+'\n'
    result.append(ost)

with open('./data/BIM_result','w') as f:
    json.dump(result, f)
    
