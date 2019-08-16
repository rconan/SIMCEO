
import sys
import dos

if __name__=="__main__":

    dospath = sys.argv[1]
    sim = dos.DOS(dospath)
    sim._run_()
