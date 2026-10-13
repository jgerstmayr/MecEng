#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Details:  Here, helpers that are conveniently called from the sample
#           script can be placed. For FlexuAI, special import functions
#           are placed here such that they do not have to be written
#           in every sample file.
#
# Author:   Sebastian Weyrer
# Date:     2026-02-19
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Try normal import first
try:
    from ExuAI.flexUtilities import GetFemObjectFromOCCGeometry, MakeFFRFObjectAndAnimateModes

except ImportError:
    # Only fix path if needed
    from pathlib import Path
    import sys

    ROOT = Path(__file__).resolve().parents[3]   # adjust depth if needed
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    # Retry import
    from ExuAI.flexUtilities import GetFemObjectFromOCCGeometry, MakeFFRFObjectAndAnimateModes
