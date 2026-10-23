
# MAIN FUNCTION TO CALL THE L1C MODULE

from l1c.src.l1c import l1c

# Directory - this is the common directory for the execution of the E2E, all modules
auxdir = r"C:\\Users\\adria\\Documents\\GitHub\\EODP\\auxiliary"
# GM dir + L1B dir
indir = r'C:\\Users\\adria\\Desktop\\carpeta\\master\\tierra\\EODP-TS-E2E\\gm_out,C:\\Users\\adria\\Desktop\\carpeta\\master\\tierra\\EODP-TS-E2E\\myoutput_l1b'
outdir = r'C:\\Users\\adria\\Desktop\\carpeta\\master\\tierra\\EODP-TS-E2E\\myoutput_l1c'
# GM dir + L1B dir


# Initialise the ISM
myL1c = l1c(auxdir, indir, outdir)
myL1c.processModule()
