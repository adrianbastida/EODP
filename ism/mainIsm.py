
# MAIN FUNCTION TO CALL THE ISM MODULE

from ism.src.ism import ism

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r"C:\\Users\\adria\\Documents\\GitHub\\EODP\\auxiliary"
indir = r"C:\\Users\\adria\\Desktop\\carpeta\\master\\tierra\\EODP-TS-E2E\\sgm_out" # small scene
outdir = r"C:\\Users\\adria\\Desktop\\carpeta\\master\\tierra\\EODP-TS-E2E\\myoutput"

# Initialise the ISM
myIsm = ism(auxdir, indir, outdir)
myIsm.processModule()
