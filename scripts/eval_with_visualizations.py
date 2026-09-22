"""Run the existing evaluator and capture seeded examples from that exact pass."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import eval_test
from src.metrics import compute_similarity_transform, joint_mesh_errors


def capture_evaluator(original, output, count, seed):
    def evaluate(name, model, loader, device, rank, world_size, bf16, steps=None):
        dataset = loader.dataset
        ids = np.random.default_rng(seed).choice(len(dataset), min(count, len(dataset)), replace=False)
        selected = {dataset.grasp_files[i].relative_to(dataset.dataset_path).with_suffix('').as_posix(): int(i) for i in ids}
        current = {}
        class Loader:
            def __len__(self):
                return len(loader)

            def __iter__(self):
                started = time.perf_counter()
                for index, batch in enumerate(loader):
                    current['batch'] = batch
                    yield batch
                    if rank == 0 and (index + 1) % 10 == 0:
                        print(f'Progress: {index + 1}/{len(loader)} batches, {time.perf_counter() - started:.0f}s', flush=True)
        class Model:
            def __getattr__(self, key):
                return getattr(model, key)
            def build_loss_dicts(self, samples, gt, **geometry):
                pred, target = model.build_loss_dicts(samples, gt, **geometry)
                batch = current['batch']
                keep = [i for i,s in enumerate(batch['stem']) if s in selected]
                if keep:
                    errors = joint_mesh_errors(pred['landmarks_3d'][keep].float(), target['landmarks_3d'][keep].float(), pred['vertices'][keep].float(), target['vertices'][keep].float())
                    folder = output / 'examples'
                    folder.mkdir(exist_ok=True, parents=True)
                    for j,i in enumerate(keep):
                        stem = batch['stem'][i]
                        np.savez_compressed(folder / f'{selected[stem]:06d}.npz',
                            stem=np.asarray(stem), index=selected[stem],
                            dataset_root=np.asarray(str(dataset.dataset_path)),
                            pred_joints=pred['landmarks_3d'][i].float().cpu().numpy(),
                            gt_joints=target['landmarks_3d'][i].float().cpu().numpy(),
                            pred_vertices=pred['vertices'][i].float().cpu().numpy(),
                            gt_vertices=target['vertices'][i].float().cpu().numpy(),
                            camera_K=batch['camera_K'][i].cpu().numpy(),
                            bbox=batch['hand_crop_bbox'][i].cpu().numpy(),
                            source_is_left=batch['source_is_left'][i].cpu().numpy(),
                            crop_rgb=batch['rgb'][i].float().cpu().numpy(),
                            **{k:float(v[j]) for k,v in errors.items()})
                return pred, target
        return original(name, Model(), Loader(), device, rank, world_size, bf16, steps=steps)
    return evaluate


def project(x, k):
    p = x @ k.T
    return p[:,:2] / np.maximum(p[:,2:],1e-6)


def skeleton(ax, xy, color, label):
    for start in (1,5,9,13,17):
        chain=[0]+list(range(start,start+4))
        ax.plot(xy[chain,0],xy[chain,1],color=color,lw=1.5)
    ax.scatter(xy[:,0],xy[:,1],s=9,c=color,label=label,zorder=3)


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from PIL import Image, ImageOps, ImageDraw
    import pickle
    output=Path(output)
    rows=[]; panels=[]
    green='#00b868';pink='#eb3d8e'
    faces=np.load('assets/mano_rhand_mesh_faces.npy')
    for file in sorted((output/'examples').glob('*.npz')):
        with np.load(file,allow_pickle=False) as z: d={k:z[k] for k in z.files}
        stem=str(d['stem']);root=Path(str(d['dataset_root']))
        with (root/f'{stem}.pkl').open('rb') as f: raw=pickle.load(f)
        rgb=cv2.cvtColor(cv2.imdecode(np.frombuffer(raw['image'],np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        pj,gj,pv,gv=[d[k] for k in ('pred_joints','gt_joints','pred_vertices','gt_vertices')]
        k=d['camera_K'];pu,gu=project(pj,k),project(gj,k)
        fig,axes=plt.subplots(1,5,figsize=(22,4.8))
        axes[0].imshow(rgb);skeleton(axes[0],gu,green,'GT');skeleton(axes[0],pu,pink,'Prediction')
        b=d['bbox'];axes[0].add_patch(plt.Rectangle(b[:2],b[2]-b[0],b[3]-b[1],fill=False,color='#f5b800',lw=1.2))
        axes[0].set_title('Full image: absolute camera projection');axes[0].set_xlim(0,rgb.shape[1]);axes[0].set_ylim(rgb.shape[0],0);axes[0].legend(fontsize=7)
        points=np.concatenate([gu,pu]);lo=points.min(0);hi=points.max(0);center=(lo+hi)/2;side=max((hi-lo).max()*1.3,80)
        for ax,verts,title,color in [(axes[1],gv,'GT mesh (native side)',green),(axes[2],pv,'Predicted mesh (unaligned)',pink)]:
            ax.imshow(rgb);uv=project(verts,k)
            ax.add_collection(PolyCollection(uv[faces],facecolors=color,edgecolors=color,alpha=.17,linewidths=.12))
            ax.set_xlim(center[0]-side/2,center[0]+side/2);ax.set_ylim(center[1]+side/2,center[1]-side/2);ax.set_title(title)
        # PA is fitted separately to joints and vertices, matching metric definitions.
        aligned_j=compute_similarity_transform(torch.tensor(pj)[None],torch.tensor(gj)[None])[0].numpy()
        aligned_v=compute_similarity_transform(torch.tensor(pv)[None],torch.tensor(gv)[None])[0].numpy()
        for ax,dim,title in [(axes[3],[0,1],'PA joints: camera X/Y (mm)'),(axes[4],[0,2],'PA mesh: camera X/Z (mm)')]:
            if ax is axes[3]:
                skeleton(ax,(gj-gj[0])[:,dim]*1000,green,'GT');skeleton(ax,(aligned_j-gj[0])[:,dim]*1000,pink,'PA prediction')
            else:
                ax.scatter(*((gv-gj[0])[:,dim]*1000).T,s=1,c=green,label='GT',alpha=.6)
                ax.scatter(*((aligned_v-gj[0])[:,dim]*1000).T,s=1,c=pink,label='PA prediction',alpha=.6)
            ax.set_title(title);ax.set_aspect('equal');ax.invert_yaxis();ax.grid(alpha=.2)
        for ax in axes[:3]:ax.axis('off')
        row=dict(stem=stem,index=int(d['index']),source_side='left' if bool(d['source_is_left']) else 'right',**{q:float(d[q]) for q in ('mpjpe','pa_mpjpe','mpvpe','pa_mpvpe')})
        fig.suptitle(f"{stem} | source={row['source_side']} | MPJPE {row['mpjpe']:.2f} / PA {row['pa_mpjpe']:.2f} | MPVPE {row['mpvpe']:.2f} / PA {row['pa_mpvpe']:.2f} mm",fontsize=11)
        fig.tight_layout();dest=output/'visualizations'/f"{int(d['index']):06d}.jpg";dest.parent.mkdir(exist_ok=True)
        fig.savefig(dest,dpi=120);plt.close(fig)
        rows.append(row);panels.append(dest)
    (output/'example_metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    board=Image.new('RGB',(1600,200*len(panels)), 'white')
    for i,p in enumerate(panels):
        im=Image.open(p);im.thumbnail((1600,200));board.paste(im,(0,i*200))
    board.save(output/'contact_sheet.jpg')
    html=['<!doctype html><meta charset="utf-8"><title>DexYCB v32 evaluation</title><style>body{font:16px sans-serif;margin:24px;background:#fafafa}img{width:100%;height:auto}section{margin-bottom:24px}pre{white-space:pre-wrap}</style><h1>DexYCB v32 checkpoint evaluation</h1><p>EMA, GT keypoint conditioning, detector crop. Green: GT; pink: prediction; yellow: detector crop. First three views are unaligned camera projections. Last two use PA for shape comparison. Examples were selected before inference with seed 20260921.</p>']
    html+=['<pre>'+json.dumps(json.loads((output/'metrics.json').read_text()),indent=2)+'</pre>']
    html += [f'<section><img loading="lazy" src="visualizations/{p.name}"></section>' for p in panels]
    (output/'index.html').write_text('\n'.join(html))
    print(f'Rendered {len(rows)} examples -> {output}',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--render',action='store_true')
    p.add_argument('--batch-size',type=int,default=128)
    a=p.parse_args()
    if a.render:render(a.output);return
    a.output.mkdir(exist_ok=True,parents=True)
    eval_test.evaluate_dataset=capture_evaluator(eval_test.evaluate_dataset,a.output,20,20260921)
    eval_test.main(config=a.config,ckpt=a.checkpoint,weights='ema',sets='dexycb_test',steps=50,batch_size=a.batch_size)
    if int(os.environ.get('RANK', '0')) == 0:
        captured = len(list((a.output / 'examples').glob('*.npz')))
        if captured != 20:
            raise RuntimeError(f'Expected 20 captured examples, got {captured}')
        render(a.output)


if __name__=='__main__':main()
