from __future__ import absolute_import

from got10k.experiments import *

from siamfc import TrackerSiamFC

from options import TestOptions


if __name__ == '__main__':
    opt = TestOptions().parse()

    # setup tracker
    net_path = 'pretrained/siamfc/model.pth'
    tracker = TrackerSiamFC(name=opt.name, weight=opt.weight, device=opt.device)

    # setup experiments
    experiments = []
    for i in range(len(opt.exps)):
        if opt.exps[i] in ['otb2013', 'OTB2013', 'OTB-2013']:
            experiments.append(ExperimentOTB('data/OTB', version=2013))
        elif opt.exps[i] in ['otb2015', 'OTB2015', 'OTB-2015']:
            experiments.append(ExperimentOTB('data/OTB', version=2015))
        elif opt.exps[i] in ['vot2018', 'VOT2018', 'VOT-2018']:
            experiments.append(ExperimentVOT('data/vot2018', version=2018))
        elif opt.exps[i] in ['got10k', 'GOT10k', 'GOT-10k']:
            experiments.append(ExperimentGOT10k('data/GOT-10k', subset='test'))
        else:
            raise NotImplementederror

    # run tracking experiments and report performance
    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
