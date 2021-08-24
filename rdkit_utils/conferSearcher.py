#!/usr/bin/env python
# J. Liu 2021.0824 
#   A Conformation Enumerator based on RDKit ETKDGv3
#     - Torsion Fingerpoint Deviation is supported in addition to RMSD
#     - I/O fileformats that not supported by RDKit will be converted by using openbabel

import sys, os
import conformers, serial

class input_params(object):
      def __init__(self, input_filename, output_filename, nconf, useTFD, rms, tfd, addH, bestRMSD, ncpu, verbose):
         self.input_filename = input_filename
         self.output_filename = output_filename
         self.nconf = nconf
         self.useTFD = (useTFD == 1)
         self.rms = rms
         self.tfd = tfd
         self.addH = (addH == 1)
         self.bestRMSD = (bestRMSD==1)
         self.ncpu = ncpu
         self.verbose = verbose
         self.convert_input_fileformat = False
         self.convert_output_fileformat = False

         if output_filename is None:
            file_name = os.path.splitext(input_filename)[0]
            self.output_filename = file_name + "_confs.sdf"
         if self.bestRMSD : self.useTFD=False
         if self.useTFD :
            self.rmsd_threshold = self.tfd
         else:
            self.rmsd_threshold = self.rms

         self.input_filename_orig = self.input_filename
         self.output_filename_orig = self.output_filename


def error(msg=""):
    print (msg)
    sys.exit(-1)

def fileformat_converter(input_filename, output_filename):
    from openbabel import pybel
    input_ext = os.path.splitext(input_filename)[1][1:]
    output_ext = os.path.splitext(output_filename)[1][1:]

    if not input_ext in pybel.informats :
        error("input file %s is not supported." % input_filename)
    if not output_ext in pybel.outformats :
        error("output file %s is not supported." % output_filename)

    output = pybel.Outputfile(output_ext, output_filename, overwrite=True)
    for mol in pybel.readfile(input_ext, input_filename):
        output.write(mol)
    output.close()

def parse_options () :
    import argparse
    #1). set arguments
    description = """Conformation Enumerator based on RDKit ETKDGv3"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-input', required=True,help="input molecule file. Required.")
    parser.add_argument('-output', help="output.")
    parser.add_argument('-nconf', default=1000, type=int, help="Maximum conformermations that will be generated. Default=1000.")
    parser.add_argument('-useTFD', default=1, type=int, help="1: use, 0: don't use. Default=1")
    parser.add_argument('-rms', default=0.5, type=float, help="RMSD threshold for identical conformers. Default=0.5")
    parser.add_argument('-tfd', default=0.02, type=float, help="TFD threshold for identical conformers. Default=0.02")
    parser.add_argument('-addH', default=0, type=int, help="1: add hydrogens using rdkit AddH(), 0: don't add hydrogens. Default=0")

    arghelp  ="1: find best RMSD for each conformer pair. This will disable useTFD. Very slow. ," 
    arghelp +="0: don't use best RMSD algorithm. " 
    arghelp +="if useTFD=False, the rmsds are calculated with aligning all conformers to the first one only. Fast."
    parser.add_argument('-bestRMSD', default=0, type=int, help=arghelp)

    parser.add_argument('-ncpu', default=0, type=int, help="the number of cpu. 0 is to use all available cpu. Default=0.")
    parser.add_argument('-verbose', default=0, type=int, help="printing more information. 0 : don't verbose, 1 : verbose when certain tasks begin, 2 : print out some data")


    #2). exit if no argument input
    if len(sys.argv) == 1 :
       parser.print_help()
       error()

    #3). output argument
    input_filename = parser.parse_args().input
    output_filename = parser.parse_args().output
    nconf = parser.parse_args().nconf
    nml = input_params( parser.parse_args().input,
                        parser.parse_args().output,
                        parser.parse_args().nconf,
                        parser.parse_args().useTFD,
                        parser.parse_args().rms,
                        parser.parse_args().tfd,
                        parser.parse_args().addH,
                        parser.parse_args().bestRMSD,
                        parser.parse_args().ncpu,
                        parser.parse_args().verbose
    )
    return (nml)

def check_fileformat(nml):
    input_ext = os.path.splitext(nml.input_filename)[1][1:]
    output_ext = os.path.splitext(nml.output_filename)[1][1:]
    if input_ext.lower() == 'smi' : nml.addH = True
    if not input_ext.lower() in ['sdf', 'mol2', 'smi', 'pkl' ] :
       nml.convert_input_fileformat = True
       file_name = os.path.splitext(nml.input_filename)[0]
       file_name += ".template.sdf"
       nml.input_filename = file_name
    if not output_ext.lower() in ['sdf', 'smi', 'pkl' ] :
       nml.convert_output_fileformat = True
       file_name = os.path.splitext(nml.output_filename)[0]
       file_name += ".out.template.sdf"
       nml.output_filename = file_name

def gen_conformers(nml):
   #1). convert input file to SDF if not supported by rdkit
   check_fileformat(nml)
   if nml.convert_input_fileformat : 
      fileformat_converter(nml.input_filename_orig, nml.input_filename)

   #2). read and convert to rdkit mol
   reader = serial.MolReader()
   mols = reader.open(nml.input_filename)
   mol = next(mols.get_mols()) # only one molecule per file

   #3). do confer search
   Searcher = conformers.ConformerGenerator(
          max_conformers  = nml.nconf,
          rmsd_threshold  = nml.rmsd_threshold,
          pool_multiplier = 5,  # max_conformers*pool_multiplier number of confers will be requested to rdkit EmbedMultipleConfs
          addH            = nml.addH,
          bestRMSD        = nml.bestRMSD,
          useTFD          = nml.useTFD,
          ncpu            = nml.ncpu,
          verbose         = nml.verbose
   )

   confs = Searcher.generate_conformers(mol)

   #4). save
   writer = serial.MolWriter()
   with writer.open(nml.output_filename,mode='w') as f:
        f.write([confs])
   if nml.convert_output_fileformat : 
      fileformat_converter(nml.output_filename, nml.output_filename_orig)

   #5). clean up
   if nml.convert_input_fileformat :  os.remove(nml.input_filename)
   if nml.convert_output_fileformat : os.remove(nml.output_filename)
    
def main():
   nml = parse_options()
   gen_conformers(nml)

if __name__ == '__main__' :
   main()
