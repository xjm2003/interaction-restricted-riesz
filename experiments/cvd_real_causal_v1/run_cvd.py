#!/usr/bin/env python3
from cvd_real import main, SafeError
import sys
if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SafeError as e:
        print('STOP: '+str(e), file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print('Interrupted; run the command again to refit.', file=sys.stderr)
        raise SystemExit(130)
