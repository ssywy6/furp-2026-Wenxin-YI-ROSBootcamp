"""Known-map grid planning used for demonstrations and navigation protection."""
import heapq
import numpy as np

def plan(model,data,destination):
    xs=np.arange(-.2,3.51,.1);ys=np.arange(-1.3,1.31,.1)
    obstacles=[(model.geom_pos[model.geom(f'obstacle_{i}').id,:2],model.geom_size[model.geom(f'obstacle_{i}').id,:2]) for i in range(2)]
    def free(p):
        xy=np.array([xs[p[0]],ys[p[1]]])
        return not any(np.all(np.abs(xy-center)<=size+np.array([.35,.45])) for center,size in obstacles)
    start=(int(np.argmin(abs(xs-data.qpos[0]))),int(np.argmin(abs(ys-data.qpos[1]))))
    end=(int(np.argmin(abs(xs-destination[0]))),int(np.argmin(abs(ys-destination[1]))))
    if not free(start):
        candidates=[(i,j) for i in range(len(xs)) for j in range(len(ys)) if free((i,j))]
        start=min(candidates,key=lambda p:(xs[p[0]]-data.qpos[0])**2+(ys[p[1]]-data.qpos[1])**2)
    queue=[(0.,start)];cost={start:0.};parent={}
    while queue:
        _,p=heapq.heappop(queue)
        if p==end:break
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            q=(p[0]+dx,p[1]+dy)
            if not 0<=q[0]<len(xs) or not 0<=q[1]<len(ys) or not free(q):continue
            new=cost[p]+np.hypot(dx,dy)
            if new<cost.get(q,np.inf):
                cost[q]=new;parent[q]=p;heapq.heappush(queue,(new+np.hypot(q[0]-end[0],q[1]-end[1]),q))
    if end not in parent:return None
    path=[end]
    while path[-1]!=start:path.append(parent[path[-1]])
    return np.array([[xs[p[0]],ys[p[1]]] for p in path[::-1]])

def follow(data,path,index):
    nearest=int(np.argmin(np.linalg.norm(path-data.qpos[:2],axis=1)));index=max(index,nearest)
    look=index;length=0.
    while look<len(path)-1 and length<.35:
        length+=np.linalg.norm(path[look+1]-path[look]);look+=1
    rot=data.xmat[1].reshape(3,3)
    local=rot[:2,:2].T@(path[look]-data.qpos[:2]);bearing=np.arctan2(local[1],local[0])
    speed=.18 if abs(bearing)<.35 else 0.
    return np.array([speed/.2-1.,np.clip(2*bearing,-.5,.5)/.5],dtype=np.float32),index

def clearance(model,xy):
    return min(np.linalg.norm(np.maximum(np.abs(xy-model.geom_pos[model.geom(f'obstacle_{i}').id,:2])-
        model.geom_size[model.geom(f'obstacle_{i}').id,:2],0.)) for i in range(2))
