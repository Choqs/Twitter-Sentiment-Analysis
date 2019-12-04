import sys
import os.path
from utils import evaluate_ei
from utils import evaluate_oc
from utils import evaluate_multilabel



def main(argv):
    """main method """

    if len(argv)!=3:
        raise ValueError('Invalid number of parameters.')


    task_type=int(argv[0])
    pred=argv[1]
    gold=argv[2]

    if(task_type==1):
        result=evaluate_ei(pred,gold)
        print ("Pearson correlation between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[0]) )
        print ("Pearson correlation for gold scores in range 0.5-1 between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[1]) )

    elif(task_type==2):
        result=evaluate_oc(pred,gold)
        print ("Pearson correlation between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[0]) )
        print ("Pearson correlation for some emotions between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[1]) )
        print ("Weighted quadratic Kappa between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[2]) )
        print ("Weighted quadratic Kappa for some emotions between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[3]) )


    else:
        result=evaluate_multilabel(pred,gold)
        print ("Multi-label accuracy (Jaccard index) between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[0]) )
        print ("Micro-averaged F1 score between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[1]) )
        print ("Macro-averaged F1 score between "+os.path.basename(pred)+" and "+os.path.basename(gold)+":\t"+str(result[2]) )



if __name__ == "__main__":
    main(sys.argv[1:])
