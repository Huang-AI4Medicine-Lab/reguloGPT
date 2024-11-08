import os

import pandas as pd

import time
import csv
import math
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import f1_score, precision_score, recall_score

import math

def load_file(finalename):
  df = pd.read_csv(finalename, encoding='ISO-8859-1')
  print(len(df))
  res = []
  edges = []
  ctxLst = []
  ctx = None
  cnt = 0
  idxLst = []
  for index, row in df.iterrows():
    if not math.isnan(row['Idx']):
      ctx = row['Context']
      edges.append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))
      
      if int(row['Idx']) not in idxLst:
        idxLst.append(int(row['Idx']))
        
    elif len(edges) > 0:
      cnt += 1
      ctxLst.append(ctx)
      res.append(edges)
      edges = []
      
  cnt += 1
  ctxLst.append(ctx)
  res.append(edges)
  
  return res, ctxLst, idxLst



def load_2file(gt_file, pr_file):
  df_gt = pd.read_csv(gt_file, encoding='ISO-8859-1')
  df_pr = pd.read_csv(pr_file, encoding='ISO-8859-1')
  print(len(df_gt), len(df_pr))
  
  gtres = {}
  prres = {}
  ctx_gt = {}  # Contexts for ground truth
  ctx_pr = {}  # Contexts for predictions
  gt_idx = set()  # Initialize as set
  pr_idx = set()  # Initialize as set
  
  # Process prediction data
  for index, row in df_pr.iterrows(): 
    idx = row['Idx']
    
    if not math.isnan(idx) and not isinstance(row['node-A'], float):
      idx = int(idx)  # Ensure idx is an integer
      pr_idx.add(idx)
      
      if idx not in prres:
        prres[idx] = []
        ctx_pr[idx] = row.get('Context', '')
        
      if row['node-B'] is None or row['node-B'] == "" or pd.isna(row['node-B']):
        print(row['Title'])
        prres[idx].append((row['node-A'].strip().lower(), row['edge'], ''))
      elif row['edge'] is None or row['edge'] == "" or pd.isna(row['edge']):
        prres[idx].append((row['node-A'].strip().lower(), '', row['node-B'].strip().lower()))
      else:
        prres[idx].append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))
        
  # Process ground truth data
  for index, row in df_gt.iterrows():
    idx = row['Idx']
    if not math.isnan(idx):
      idx = int(idx)
      if idx in pr_idx:  # Only add if index is also in predictions
        gt_idx.add(idx)
        
        if idx not in gtres:
          gtres[idx] = []
          ctx_gt[idx] = row.get('Context', '')
          
        if row['edge'] is None or row['edge'] == "" or pd.isna(row['edge']):
          gtres[idx].append((row['node-A'].strip().lower(), '', row['node-B'].strip().lower()))
        else:
          gtres[idx].append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))
        
  # Ensure only indices in both gt and pr are processed
  common_indices = gt_idx & pr_idx
  gtLst = [gtres[idx] for idx in common_indices]
  preLst = [prres[idx] for idx in common_indices]
  ctx_gt_list = [ctx_gt[idx] for idx in common_indices]  # Extract contexts for ground truth
  ctx_pr_list = [ctx_pr[idx] for idx in common_indices]  # Extract contexts for predictions
  
  return gtLst, ctx_gt_list, list(common_indices), preLst, ctx_pr_list, list(common_indices)


# getting all the nodes using list of triplets of each title
def get_node(edgelst):
  nodeSet = set()
  for edge in edgelst:
    nodeSet.add(edge[0].strip().lower())
    nodeSet.add(edge[2].strip().lower())
  return nodeSet


def fuzzyMatch(node, gtNodes, threshold=0.5):
  for itm in gtNodes:
    if (node in itm) or (itm in node):
      return itm

    nLst = node.split(' ') 
    itmLst = itm.split(' ') 
    union = list(set(nLst).union(set(itmLst)))
    intersection = list(set(nLst).intersection(set(itmLst)))
    if len(intersection) / len(union) >= threshold:
      return itm
  
  return node




def node_mapping(gtLst, preLst):
  gtNodeLst = []
  prNodeLst = []
  new_preLst = []
  
  for idx, (gt, pre) in enumerate(zip(gtLst, preLst)): #zip triplets of 400 titles between gt and pre (400 each)
    gtNodes = get_node(gt) # ground truth nodes in a title
    gtNodeLst.append(gtNodes) # list of all ground truth nodes from 400 titles
    
    prNodes = get_node(pre) # list of predicted nodes in the same title
    
    normDic = {}
    new_pre_Node = []
    
    for node in prNodes:
      if node in gtNodes: #if predicted node is one of ground-truth node, keep it
        normDic[node] = node
      else:
        normDic[node] = fuzzyMatch(node, gtNodes) #otherwise, find the best matching (threshold = 0.5) or keep the original node if it fails
        
      new_pre_Node.append(normDic[node])
      
    preEdgeLst = []
    for edge in pre:
      newEdge = (normDic[edge[0]], edge[1], normDic[edge[2]]) # replace the raw nodes with the normalized nodes in each triplet
      preEdgeLst.append(newEdge)
      
    prNodeLst.append(new_pre_Node) #normalized nodes
    new_preLst.append(preEdgeLst) #triplets with the normalized nodes
    
  return gtNodeLst, prNodeLst, gtLst, new_preLst #list of nodes and list of triplets


def getAcc(ctx_gt, ctx_pr):
  assert len(ctx_gt) == len(ctx_pr)
  cnt = 0
  
  for idx, (itm1, itm2) in enumerate(zip(ctx_gt, ctx_pr)):
    # print(idx, itm1, itm2)
    
    if itm1 is None or itm2 is None or itm1 == "" or itm2 == "" or (isinstance(itm1, float) and math.isnan(itm1)) or (isinstance(itm2, float) and math.isnan(itm2)):
      continue
    
    #make context into lowercase letters
    str1 = itm1.strip().lower() #ground truth
    str2 = itm2.strip().lower() #prediction
     
    if (str1 in str2) or (str2 in str1) or (str2[0:-1] in str1) or (str2 in (str1 + " cells")) or (str2 in (str1 + " stem cells")):
      cnt += 1
      
  return cnt / len(ctx_gt)




##############################################################################




gt_file = 'benchmark_ground_truth.csv'   # groundtruth
pr_file = 'reguloGPT_baseline_prediction.csv'       # your predicted KG!

gtLst, ctx_gt, gt_idx, preLst, ctx_pr, pr_idx = load_2file(gt_file, pr_file)
print(len(gtLst), len(preLst))
print(len(ctx_gt), len(ctx_pr))
print(len(gt_idx), len(pr_idx))
assert len(gtLst) == len(preLst), "title # should be the same"

gt_node, pre_node, gtLst, preLst = node_mapping(gtLst, preLst)

# context accuracy
ctxscore = getAcc(ctx_gt, ctx_pr)
getAcc(ctx_gt, ctx_pr)
nodescore = score(gt_node, pre_node)
edgescore = score(gtLst, preLst)
      
node = pd.DataFrame([nodescore])
edge = pd.DataFrame([edgescore])
context = pd.DataFrame({'acc': [ctxscore]})
      
# Create the final DataFrame
output = pd.DataFrame(columns=['node.f1', 'node.re', 'node.pr', 'edge.f1', 'edge.re', 'edge.pr', 'context.acc'])

final_df = pd.DataFrame({
  'node.f1': node['f1'],
  'node.re': node['recall'],
  'node.pr': node['precision'],
  'edge.f1': edge['f1'],
  'edge.re': edge['recall'],
  'edge.pr': edge['precision'],
  'context.acc': context['acc']
  })
output = pd.concat([output, final_df], ignore_index=True)

fid = f"reguloGPT_results.xlsx"
output.to_excel(fid)
