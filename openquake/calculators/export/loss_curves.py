#  -*- coding: utf-8 -*-
#  vim: tabstop=4 shiftwidth=4 softtabstop=4

#  Copyright (c) 2017, GEM Foundation

#  OpenQuake is free software: you can redistribute it and/or modify it
#  under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  OpenQuake is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.

#  You should have received a copy of the GNU Affero General Public License
#  along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.
import numpy
from openquake.baselib.python3compat import decode
from openquake.commonlib import writers


class LossCurveExporter(object):
    """
    Exporter for the loss curves. The most important method is
    `.export(export_type, what)` where `export_type` is a string like 'csv',
    and `what` is a string called export specifier. Here are some examples
    for the export specifier:

    sid-42/rlzs   # export loss curves of site #42 for all realizations
    sid-42/rlz-003   # export all loss curves of site #42, realization #3
    sid-42/stats   # export statistical loss curves of site #42
    sid-42/mean   # export mean loss curves of site #42
    sid-42/quantile-0.1   # export quantile loss curves of site #42

    ref-a1/rlzs  # export loss curves of asset a1 for all realizations
    ref-a1/rlz-003   # export loss curves of asset a1, realization 3
    ref-a1/stats     # export statistical loss curves of asset a1
    ref-a1/mean     # export mean loss curves of asset a1
    ref-a1/quantile-0.1    # export quantile loss curves of asset a1
    """
    def __init__(self, dstore):
        oq = dstore['oqparam']
        self.dstore = dstore
        self.assetcol = dstore['assetcol']
        arefs = [decode(aref) for aref in self.dstore['asset_refs']]
        self.str2asset = {arefs[asset.idx]: asset for asset in self.assetcol}
        self.asset_refs = self.dstore['asset_refs'].value
        self.loss_types = dstore.get_attr('composite_risk_model', 'loss_types')
        self.R = len(dstore['realizations'])
        self.stats = ['mean'] + [
            'quantile-%s' % q for q in oq.quantile_loss_curves]
        self.S = len(self.stats)
        self.stat2idx = {stat: i for i, stat in enumerate(self.stats)}

    def parse(self, what):
        """
        :param what:
            can be 'rlz-1/ref-asset1', 'rlzs/sid-2', ...
        """
        kind, spec = what.split('/')
        if not spec.startswith(('ref-', 'sid-')):
            raise ValueError('Wrong specification in %s' % what)
        if not (kind in ('rlzs', 'stats', 'mean') or kind.startswith(('rlz-'))
                or kind.startswith('quantile-')):
            raise ValueError('Wrong export key in %s' % what)
        if spec.startswith('sid-'):  # passed the site ID
            sid = int(spec[4:])
            aids = []
            arefs = []
            for aid, rec in enumerate(self.assetcol.array):
                if rec['site_id'] == sid:
                    aids.append(aid)
                    arefs.append(self.asset_refs[rec['idx']])
        else:  # passed the asset name
            arefs = [spec[4:]]
            aids = [self.str2asset[arefs[0]].ordinal]
        return aids, arefs, spec, kind

    def export_csv(self, kind, spec, asset_refs, curves_dict):
        """
        :param kind: kind of curves to export (rlzs, stats)
        :param spec: specific asset or site to export
        :param asset_refs: names of the assets
        :param curves_dict: a dictionary tag -> loss curves
        """
        writer = writers.CsvWriter(fmt=writers.FIVEDIGITS)
        for key in sorted(curves_dict):
            recs = curves_dict[key]
            data = [['asset', 'loss_type', 'loss', 'poe']]
            for loss_type in self.loss_types:
                array = recs[loss_type]
                for aref, losses, poes in zip(
                        asset_refs, array['losses'], array['poes']):
                    for loss, poe in zip(losses, poes):
                        data.append((aref, loss_type, loss, poe))
            dest = self.dstore.build_fname(
                'loss_curves', '%s-%s' % (spec, key), 'csv')
            writer.save(data, dest)
        return writer.getsaved()

    def export_npz(self, kind, spec, asset_refs, curves_dict):
        """
        :param kind: kind of curves to export (rlzs, stats)
        :param spec: specific asset or site to export
        :param asset_refs: names of the assets
        :param curves_dict: a dictionary tag -> loss curves
        """
        dest = self.dstore.build_fname('loss_curves-' + kind, spec, 'npz')
        numpy.savez_compressed(dest, asset_refs=asset_refs, **curves_dict)
        return [dest]

    def export(self, export_type, what):
        """
        :param export_type: 'csv', 'json', ...
        :param what: string describing what to export
        :returns: list of exported file names
        """
        aids, arefs, spec, kind = self.parse(what)
        curves = self.export_curves(aids, kind)
        return getattr(self, 'export_' + export_type)(
            kind, spec, arefs, curves)

    def export_curves(self, aids, kind):
        """
        :returns: a dictionary kind -> record of dtype loss_curve_dt
        """
        if kind.startswith('rlz'):  # individual outputs
            data = self.dstore['loss_curves-rlzs'][aids]  # shape (A, R)
            if kind == 'rlzs':
                return {'rlz-%03d' % rlzi: data[:, rlzi]
                        for rlzi in range(self.R)}
            else:
                rlzi = int(kind[4:])
                return {kind: data[:, rlzi]}
        else:  # statistical outputs
            data = self.dstore['loss_curves-stats'][aids]  # shape (A, R)
            if kind == 'stats':
                return {self.stats[s]: data[:, s] for s in range(self.S)}
            else:
                s = self.stat2idx[kind]
                return {kind: data[:, s]}

        # otherwise event_based
        raise NotImplementedError
