from experiments_vfa.vfa.datasets.pairwise_dataset import PairwiseDataset

class LDM100kDataset(PairwiseDataset):
    def __init__(self, configs, params):
        super().__init__(configs, params)

    def __getitem__(self, k):
        sample = super().__getitem__(k)

        return sample

    # def export(self, results, sample):
    #     if self.params['save_results'] == 0:
    #         return None
    #
    #     super().export(results, sample)
    #
    #     '''for L2R 2024 LUMIR submission'''
    #     disp = results['grid'] - identity_grid_like(results['grid'], normalize=False)
    #     disp = disp.detach().cpu().numpy()[0].transpose(1, 2, 3, 0)
    #
    #     submission_path = pathlib.Path(self.params['output_dir']) / 'experiments' / 'l2r2024lumir' / 'submission'
    #     submission_path.mkdir(exist_ok=True, parents=True)
    #
    #     ref_obj = nib.load(sample['f_img_path'][0])
    #     nib.save(nib.Nifti1Image(disp, ref_obj.affine), str(submission_path / sample['filename'][0]))
    #
