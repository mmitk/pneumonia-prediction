def bal_acc(tp, fn, tn, fp):
    return (tp/(tp+fn)+(tn/(tn+fp)))/2.0
    

while(True):
    tp = float(input('tp: '))
    fn = float(input('fn: '))
    tn = float(input('tn: '))
    fp = float(input('fp: '))
    print(bal_acc(tp,fn,tn,fp))