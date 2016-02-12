
import os.path as osp
import sys
import getopt
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import json


sys.path.insert(1, 'utils/COCO-python-wrapper/coco-master/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
def main(argv):
	## Parsing the command	
	in_path = ''
	out_path = ''
	ann_path = ''
	try:
		opts, args = getopt.getopt(argv,"hi:o:a:",["in=","out=","annotation="])
	except getopt.GetoptError:
		print 'test.py -i <inputfile> -o <outputfile> -a <annotationfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'test.py -i <inputfile> -o <outputfile> -a <annotationfile>'
			sys.exit()
		elif opt in ("-i", "--in"):
			in_path = arg
		elif opt in ("-o", "--out"):
			out_path = arg
		elif opt in ("-a", "--annotation"):
			ann_path = arg
	print('Performing evaluation using Coco Python API...')
	_COCO = COCO(ann_path)
	_cats = _COCO.loadCats(_COCO.getCatIds())
	_classes = tuple(['__background__'] + [c['name'] for c in _cats])
	_do_eval(in_path,out_path, _COCO, _classes)
def _print_eval_metrics(coco_eval,classes):
	## The function is borrowed from https://github.com/rbgirshick/fast-rcnn/ and changed
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print ('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh)
        print '{:.1f}'.format(100 * ap_default)
        for cls_ind, cls in enumerate(classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print '{:.1f}'.format(100 * ap)

        print '~~~~ Summary metrics ~~~~'
        coco_eval.summarize()

def _do_eval(res_file, output_dir,_COCO,classes):
## The function is borrowed from https://github.com/rbgirshick/fast-rcnn/ and changed
        ann_type = 'bbox'
        coco_dt = _COCO.loadRes(res_file)
        coco_eval = COCOeval(_COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        _print_eval_metrics(coco_eval,classes)
        # Write the result file
        eval_file = osp.join(output_dir)
        eval_result = {}
        eval_result['precision'] = coco_eval.eval['precision']
        eval_result['recall'] = coco_eval.eval['recall']
        sio.savemat(eval_file,eval_result)
        print 'Wrote COCO eval results to: {}'.format(eval_file)

if __name__ == "__main__":
   main(sys.argv[1:])