import sys 
sys.path.append('./././')
from eval_lib.detect_eval.structure.measurers import QuadMeasurer
from eval_lib.detect_eval.validDataset import ImageValDataset
import torch.utils.data as data
import argparse



def eval_main():
    parser = get_args()  
    gt_root = parser.gt_root
    pred_root = parser.pred_root
    measurer = QuadMeasurer()
    test_dataset = ImageValDataset( gt_root, pred_root) 
    dataloader = data.DataLoader(test_dataset, batch_size=1)
    raw_metrics = []
    for index, batch in enumerate(dataloader):
        # polygons 
        polygons = batch['polygons'] # [1, 3, 4, 2]
        polys = []
        for p_b in polygons:
            polys_temp = []
            for p in p_b:
                polys_temp.append(p.numpy())
            polys.append(polys_temp)
        batch['polygons'] = polys  
        ignore = batch['ignore_tags']
        ignores = []
        for ig in ignore:
            ignores.append(ig)
        batch['ignore_tags'] = ignores
        outputs = []
        scores = []
        for b in batch['box_pred']:# batch 
            outputs.append(b.numpy())
        
        for s in batch['score_pred']:
            scores.append(s.numpy())
        output = outputs, scores
        raw_metric = measurer.validate_measure(batch, output, is_output_polygon=False, box_thresh=0.5)
        raw_metrics.append(raw_metric)
    
    metrics = measurer.gather_measure(raw_metrics)
    for key, metric in metrics.items():
        print('%s : %f (%d)' % (key, metric.avg, metric.count))

def get_args():
    parser = argparse.ArgumentParser(description='Handwriting detection task evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gt_root', type=str, required=True, default=None, help='the directory gt ' ) 
    parser.add_argument('-p', '--pred_root', type=str, required=True, default=None, help='the directory pred' ) 
    return parser.parse_args()

if __name__ == '__main__':
    eval_main() 





