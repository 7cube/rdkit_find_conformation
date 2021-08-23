"""
Conformer generation.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
# J.Liu 2021.0822 : import TDF module
from rdkit.Chem import TorsionFingerprints


class ConformerGenerator(object):
    """
    Generate molecule conformers.

    Procedure
    ---------
    1. Generate a pool of conformers.
    2. Minimize conformers.
    3. Prune conformers using an RMSD threshold.

    Note that pruning is done _after_ minimization, which differs from the
    protocol described in the references.

    References
    ----------
    * http://rdkit.org/docs/GettingStartedInPython.html
      #working-with-3d-molecules
    * http://pubs.acs.org/doi/full/10.1021/ci2004658

    Parameters
    ----------
    max_conformers : int, optional (default 1)
        Maximum number of conformers to generate (after pruning).
    rmsd_threshold : float, optional (default 0.5)
        RMSD threshold for pruning conformers. If None or negative, no
        pruning is performed.
    force_field : str, optional (default 'uff')
        Force field to use for conformer energy calculation and
        minimization. Options are 'uff', 'mmff94', and 'mmff94s'.
    pool_multiplier : int, optional (default 10)
        Factor to multiply by max_conformers to generate the initial
        conformer pool. Since conformers are pruned after energy
        minimization, increasing the size of the pool increases the chance
        of identifying max_conformers unique conformers.
    """
    # J.Liu 2021.0822: 
    #     -  add verbose to print more information when debugging
    #             * verbose = 0 : don't verbose
    #                         1 : verbose when certain tasks begin
    #                         2 : print out some data
    #     -  add 'addH' flag to optionally add hydrogens
    #     -  add bestRMSD flag to optionally choose rmsd methods
    #             * True : each conformer pair align each other and find the best rmsd, very slow
    #             * False: every conformers align to the first one if useTFD=False. very fast
    #     -  add useTFD flag to calculate TFD instead of RMSD. 
    #             * torsion fingerprint deviation. no need to align
    #             * https://pubs.acs.org/doi/full/10.1021/ci2002318
    def __init__(self, max_conformers=1, rmsd_threshold=0.5, force_field='uff',
                 pool_multiplier=10,
                 addH=False, bestRMSD=False, useTFD=True,verbose=0):
        self.max_conformers = max_conformers
        if rmsd_threshold is None or rmsd_threshold < 0:
            rmsd_threshold = -1.
        self.rmsd_threshold = rmsd_threshold
        self.force_field = force_field

        self.pool_multiplier = pool_multiplier
        self.verbose = verbose
        self.addH = addH
        self.bestRMSD = bestRMSD
        self.useTFD = useTFD
        if useTFD and rmsd_threshold == 0.5 :
           self.rmsd_threshold = 0.02  # it is 0.2 in TFD paper

    def __call__(self, mol):
        """
        Generate conformers for a molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        return self.generate_conformers(mol)

    def generate_conformers(self, mol):
        """
        Generate conformers for a molecule.

        This function returns a copy of the original molecule with embedded
        conformers.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """

        # initial embedding
        if self.verbose > 0 : print('Generating conformations ..')
        mol = self.embed_molecule(mol)
        if not mol.GetNumConformers():
            msg = 'No conformers generated for molecule'
            if mol.HasProp('_Name'):
                name = mol.GetProp('_Name')
                msg += ' "{}".'.format(name)
            else:
                msg += '.'
            raise RuntimeError(msg)

        # minimization and pruning
        if self.verbose > 0 : print('Minimizing conformations ..')
        self.minimize_conformers(mol)
        if self.verbose > 0 : print('Pruning conformations ..')
        mol = self.prune_conformers(mol)

        return mol

    def embed_molecule(self, mol):
        """
        Generate conformers, possibly with pruning.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        # J. Liu 2021.0822 : 
        #       - set ETKDGv3 for the support of macrocycle molecules
        #       - activate pruneRmsThresh to remove highly similar conformations
        params = AllChem.ETKDGv3()
        params.useBasicKnowledge = True
        params.useExpTorsionAnglePrefs = True
        params.useMacrocycleTorsions = True
        params.useSmallRingTorsions = True
        params.enforceChirality = True
        params.numThreads = 0
        params.pruneRmsThresh = 0.2
        params.maxIterations = 1000

        if self.addH : mol = Chem.AddHs(mol)  # add hydrogens
        n_confs = self.max_conformers * self.pool_multiplier
        AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        return mol

    def get_molecule_force_field(self, mol, conf_id=None, **kwargs):
        """
        Get a force field for a molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        conf_id : int, optional
            ID of the conformer to associate with the force field.
        kwargs : dict, optional
            Keyword arguments for force field constructor.
        """
        if self.force_field == 'uff':
            ff = AllChem.UFFGetMoleculeForceField(
                mol, confId=conf_id, **kwargs)
        elif self.force_field.startswith('mmff'):
            AllChem.MMFFSanitizeMolecule(mol)
            mmff_props = AllChem.MMFFGetMoleculeProperties(
                mol, mmffVariant=self.force_field)
            ff = AllChem.MMFFGetMoleculeForceField(
                mol, mmff_props, confId=conf_id, **kwargs)
        else:
            raise ValueError("Invalid force_field " +
                             "'{}'.".format(self.force_field))
        return ff

    def minimize_conformers(self, mol):
        """
        Minimize molecule conformers.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        for conf in mol.GetConformers():
            ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
            ff.Minimize()

    def get_conformer_energies(self, mol):
        """
        Calculate conformer energies.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.

        Returns
        -------
        energies : array_like
            Minimized conformer energies.
        """
        energies = []
        for conf in mol.GetConformers():
            ff = self.get_molecule_force_field(mol, conf_id=conf.GetId())
            energy = ff.CalcEnergy()
            energies.append(energy)
        energies = np.asarray(energies, dtype=float)
        return energies

    def prune_conformers(self, mol):
        """
        Prune conformers from a molecule using an RMSD threshold, starting
        with the lowest energy conformer.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.

        Returns
        -------
        A new RDKit Mol containing the chosen conformers, sorted by
        increasing energy.
        """
        if self.rmsd_threshold < 0 or mol.GetNumConformers() <= 1:
            return mol
        if self.verbose > 0 : print("calculating energies ..")
        energies = self.get_conformer_energies(mol)

        if self.verbose > 0 : print("calculating rmsd ..")
        if self.bestRMSD :
             rmsd = self.get_conformer_rmsd(mol)
        elif self.useTFD :
             rmsd = self.get_rmsd_matrix(mol,'TFD')
        else:
             rmsd = self.get_rmsd_matrix(mol,'RMSD')

        sort = np.argsort(energies)  # sort by increasing energy
        keep = []  # always keep lowest-energy conformer
        discard = []
        for i in sort:

            # always keep lowest-energy conformer
            if len(keep) == 0:
                keep.append(i)
                continue

            # discard conformers after max_conformers is reached
            if len(keep) >= self.max_conformers:
                discard.append(i)
                continue

            # get RMSD to selected conformers
            this_rmsd = rmsd[i][np.asarray(keep, dtype=int)]

            # discard conformers within the RMSD threshold
            if np.all(this_rmsd >= self.rmsd_threshold):
                keep.append(i)
            else:
                discard.append(i)

        if self.verbose > 1 : 
            import pandas as pd
            rmsd_df = pd.DataFrame(rmsd)
            print("energies are:\n",energies)
            print("rmsd matrix is\n", rmsd_df)
            print("energies id after sorting", sort)
            print("retained confs :", keep)
            print("total confs retained :", len(keep))
        # create a new molecule to hold the chosen conformers
        # this ensures proper conformer IDs and energy-based ordering
        new = Chem.Mol(mol)
        new.RemoveAllConformers()
        conf_ids = [conf.GetId() for conf in mol.GetConformers()]
        for idx,i in enumerate(keep):
            conf = mol.GetConformer(conf_ids[i])
            new.AddConformer(conf, assignId=True)
        self.align_conformers(new)
        return new

    def align_conformers(self, mol):
        rmslist = []
        AllChem.AlignMolConformers(mol, RMSlist=rmslist)
        if self.verbose > 1 :
           print("Align the final conformers and their rmsd are: ", rmslist)
        return 

    def get_rmsd_matrix(self, mol, mode="TFD"):
        if mode == "TFD":
                dmat = TorsionFingerprints.GetTFDMatrix(mol)
        else:
                dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
        N = mol.GetNumConformers()
        rmsd = np.zeros((N,N))
        idx = 0
        for j in range(N):
            for i in range(N):
                if i >= j: continue
                rmsd[i,j] = dmat[idx]
                rmsd[j,i] = rmsd[i,j]
                idx += 1
        return rmsd

    @staticmethod
    def get_conformer_rmsd(mol):
        """
        Calculate conformer-conformer RMSD.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        rmsd = np.zeros((mol.GetNumConformers(), mol.GetNumConformers()),
                        dtype=float)
        for i, ref_conf in enumerate(mol.GetConformers()):
            for j, fit_conf in enumerate(mol.GetConformers()):
                if i >= j:
                    continue
                rmsd[i, j] = AllChem.GetBestRMS(mol, mol, ref_conf.GetId(),
                                                fit_conf.GetId())
                rmsd[j, i] = rmsd[i, j]
        return rmsd
