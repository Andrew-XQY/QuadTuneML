"""Scientific plot contracts: font scaling, validation-only data and source matching."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

from plot_paper import (TARGETS, aggregate_efficiency, plot_parity, plot_efficiency,
                        sha256, variability_widths, validate_knee_summary)


def fixture():
    rows=[]
    for target in TARGETS:
        for n in [500, 1000]:
            for seed, factor in [(42, .8), (43, 1.2)]:
                rows.append(dict(training_samples=n, target=target, seed=seed,
                    validation_scaled_MAE=.06*factor, validation_scaled_RMSE=.08*factor,
                    validation_scaled_R2=.8*factor, scaled_MAE=999, scaled_RMSE=999, scaled_R2=-999))
    protocol=dict(completed=True, selection_split='validation',repeat_seeds=[42,43],training_counts=[500,1000])
    return pd.DataFrame(rows),protocol


class PaperPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_every_text_size_doubles_for_each_figure(self):
        frame,protocol=fixture()
        agg=aggregate_efficiency(frame,protocol)
        preds=pd.DataFrame({f'{kind}_scaled_{t}':[.1,.5,.9] for t in TARGETS for kind in ['true','pred']})
        metrics={'scaled':{t:{'MAE':.02,'R2':.9} for t in TARGETS}}
        for plot in ['parity','efficiency']:
            fonts=[]
            for scale in [1.,2.]:
                options=dict(font_scale=scale,figsize=[12,5],wspace=.3)
                fig=plot_parity(preds,metrics,options) if plot=='parity' else plot_efficiency(agg,options)
                fig.canvas.draw()
                fonts.append(sorted({item.get_fontsize() for item in fig.findobj(Text)}))
                self.assertFalse(any(item.get_text() in ['(a)','(b)'] for item in fig.findobj(Text)))
            np.testing.assert_allclose(fonts[1],np.array(fonts[0])*2)

    def test_validation_means_sd_ignore_any_test_columns(self):
        frame,protocol=fixture()
        agg=aggregate_efficiency(frame,protocol)
        self.assertAlmostEqual(agg.loc[('emittance_x',500),('validation_scaled_MAE','mean')],.06)
        self.assertAlmostEqual(agg.loc[('emittance_x',500),('validation_scaled_MAE','std')],np.std([.048,.072],ddof=1))
        frame[['scaled_MAE','scaled_RMSE','scaled_R2']]=0
        pd.testing.assert_frame_equal(agg,aggregate_efficiency(frame,protocol))
        for bad in [frame.iloc[:-1],pd.concat([frame,frame.iloc[:1]])]:
            with self.assertRaises(ValueError):aggregate_efficiency(bad,protocol)
        with self.assertRaises(ValueError):aggregate_efficiency(frame,{**protocol,'completed':False})
        with self.assertRaises(ValueError):aggregate_efficiency(frame,{**protocol,'selection_split':'test'})

    def test_right_only_combined_legend_and_supported_markers(self):
        frame,protocol=fixture()
        knee={'targets':{t:{'display_knee': t=='emittance_x','knee':500} for t in TARGETS}}
        fig=plot_efficiency(aggregate_efficiency(frame,protocol),dict(font_scale=1.3,figsize=[11.2,4.5],wspace=.36),knee)
        fig.canvas.draw()
        self.assertIsNone(fig.axes[0].get_legend())
        labels=[text.get_text() for text in fig.axes[1].get_legend().get_texts()]
        self.assertEqual(labels,['MAE','RMSE',r'$R^2$','Estimated knee'])
        self.assertEqual(len(fig.axes[0].lines),3) # MAE, RMSE, supported knee
        self.assertEqual(len(fig.axes[1].lines),2) # no unsupported knee

    def test_variability_and_knee_provenance_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.json';path.write_text('{}')
            summary=dict(status='complete',preprocessing={'sha256':sha256(path)},
                         spaces={'scaled':{t:{'half_max_observed_range':.02} for t in TARGETS}})
            self.assertEqual(variability_widths(summary,path),{t:.02 for t in TARGETS})
            for change in [{'status':'running'},{'preprocessing':{'sha256':'wrong'}}]:
                with self.assertRaises(ValueError):variability_widths({**summary,**change},path)
            valid={'sweep_sha256':sha256(path)}
            self.assertEqual(validate_knee_summary(valid,path),valid)
            with self.assertRaises(ValueError):validate_knee_summary({'sweep_sha256':'wrong'},path)


if __name__=='__main__':unittest.main()
