#!/usr/bin/env python3
import sys
from run import main
if __name__ == '__main__':
    main(['--dataset','cvd',*sys.argv[1:]])
