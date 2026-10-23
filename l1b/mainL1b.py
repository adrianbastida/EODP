
# MAIN FUNCTION TO CALL THE L1B MODULE

from l1b.src.l1b import l1b

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r"C:\\Users\\adria\\Documents\\GitHub\\EODP\\auxiliary"
indir = r"C:\\Users\\adria\\Desktop\\carpeta\\master\\tierra\\EODP-TS-E2E\\myoutput"
outdir = r"C:\\Users\\adria\\Desktop\\carpeta\\master\\tierra\\EODP-TS-E2E\\myoutput_l1b"

# Initialise the ISM
myL1b = l1b(auxdir, indir, outdir)
myL1b.processModule()
