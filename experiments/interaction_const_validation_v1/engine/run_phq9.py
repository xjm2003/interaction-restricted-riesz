#!/usr/bin/env python3
import sys
from run import main
if __name__ == '__main__':
    main(['--dataset','phq9',*sys.argv[1:]])
