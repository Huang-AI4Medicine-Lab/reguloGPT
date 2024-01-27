import openai
import os

import pandas as pd

import time
import csv
import math
from sklearn.metrics import precision_recall_fscore_support as prfs
# from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, recall_score, precision_score
def load_file(finalename):

	df = pd.read_csv(finalename, encoding='ISO-8859-1')
	print(len(df))
	res = []
	edges = []
	ctxLst = []
	ctx = None
	cnt = 0
	idxLst = []
	cntedge = 0
	for index, row in df.iterrows():

		if not math.isnan(row['Idx']):
			cntedge += 1
			ctx = row['Context']
			edges.append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))

			if int(row['Idx']) not in idxLst:
				# if len(idxLst) != len(ctxLst):
				# 	print(int(row['Idx']))
				idxLst.append(int(row['Idx']))
			# Set.add(int(row['index']))
		# elif cnt == 3:
		# 	break
		elif len(edges) > 0:
			cnt += 1
			ctxLst.append(ctx)
			res.append(edges)
			edges = []

	print('cntedge', cntedge)
	cnt += 1
	ctxLst.append(ctx)
	res.append(edges)

	return res, ctxLst, idxLst

def load_file2(finalename, gt_idx):

	df = pd.read_csv(finalename, encoding='ISO-8859-1')
	print(len(df))
	res = []
	idxLst = []

	edges = {}
	ctxLst = ['' for itm in gt_idx]
	cnt = 0
	for index, row in df.iterrows():
		idx = row['Idx']
		if idx not in edges:
			edges[idx] = []
		edges[idx].append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))

	for idx in gt_idx:
		if idx not in edges:
			cnt += 1
			res.append([])
		else:
			res.append(edges[idx]) 
		
		idxLst.append(idx)
	print("missing", cnt)

	return res, ctxLst, idxLst


def load_2file(gt_file, pr_file):
	df_gt = pd.read_csv(gt_file, encoding='ISO-8859-1')
	df_pr = pd.read_csv(pr_file, encoding='ISO-8859-1')
	print(len(df_gt), len(df_pr))
	gtres = {}
	prres = {}

	gt_idx = set()
	pr_idx = set()

	for index, row in df_pr.iterrows():
		idx = row['Idx']
		pr_idx.add(idx)
		if idx not in prres:
			prres[idx] = []
		prres[idx].append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))

	for index, row in df_gt.iterrows():
		if not math.isnan(row['Idx']):
			if row['Idx'] not in pr_idx:
				continue
			gt_idx.add(row['Idx'])
			if row['Idx'] not in gtres:
				gtres[row['Idx']] = []
			gtres[row['Idx']].append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))

	gtLst = []
	preLst= []
	assert len(pr_idx) == len(gt_idx)
	for idx in pr_idx:
		gtLst.append(gtres[idx])
		preLst.append(prres[idx])

	return gtLst, [''], list(gt_idx), preLst, [''], list(pr_idx)

	###########


	# for index, row in df_pr.iterrows():
	# 	# if not math.isnan(row['Idx']):
	# 	# 	pr_idx.add(row['Idx'])
	# 	# 	if row['Idx'] not in prres:
	# 	# 		prres[row['Idx']] = []
	# 	# 	if not isinstance(row['node-A'], float):
	# 	# 		prres[row['Idx']].append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))
	# 	###
	# 	if not math.isnan(row['Idx']) and not isinstance(row['node-A'], float):
	# 		pr_idx.add(row['Idx'])
	# 		if row['Idx'] not in prres:
	# 			prres[row['Idx']] = []
	# 		prres[row['Idx']].append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))



	# for index, row in df_gt.iterrows():
	# 	if not math.isnan(row['Idx']):
	# 		if row['Idx'] not in pr_idx:
	# 			continue
	# 		gt_idx.add(row['Idx'])
	# 		if row['Idx'] not in gtres:
	# 			gtres[row['Idx']] = []
	# 		gtres[row['Idx']].append((row['node-A'].strip().lower(), row['edge'].strip().lower(), row['node-B'].strip().lower()))


	# gtLst = []
	# preLst= []
	# assert len(pr_idx) == len(gt_idx)
	# for idx in pr_idx:
	# 	gtLst.append(gtres[idx])
	# 	preLst.append(prres[idx])

	# return gtLst, [], list(gt_idx), preLst, [], list(pr_idx)

def get_node(edgelst):
	nodeSet = set()
	for edge in edgelst:
		nodeSet.add(edge[0].strip().lower())
		nodeSet.add(edge[2].strip().lower())
	return nodeSet

def fuzzyMatch(node, gtNodes, threshold = 0.5):
	for itm in gtNodes:
		if (node in itm) or  (itm in node):
		# if (node in itm):
			return itm 
		nLst =  node.split(' ')
		itmLst = itm.split(' ')
		union = list( set(nLst).union(set(itmLst)) )
		intersection = list( set(nLst).intersection(set(itmLst)) )

		if len(intersection) / len(union) >= threshold:
			return itm

	return node


def node_normalization(gtLst, preLst):
	gtNodeLst = []
	prNodeLst = []
	new_preLst = []

	for idx, (gt, pre) in enumerate(zip(gtLst, preLst)):
		gtNodes = get_node(gt)
		# print(gtNodes)
		gtNodeLst.append(gtNodes)

		prNodes = get_node(pre)
		# print(prNodes)

		normDic = {}
		new_pre_Node = []

		for node in prNodes:
			if node in gtNodes:
				normDic[node] = node
			else:
				normDic[node] = fuzzyMatch(node, gtNodes)
			
			new_pre_Node.append(normDic[node])

		preEdgeLst = []
		for edge in pre:
			# print(edge)
			newEdge = (normDic[edge[0]], edge[1], normDic[edge[2]])
			# print(newEdge)
			preEdgeLst.append(newEdge)

		prNodeLst.append(new_pre_Node)
		new_preLst.append(preEdgeLst)


	return gtNodeLst, prNodeLst, gtLst, new_preLst

def getAcc(ctx_gt, ctx_pr):
	assert len(ctx_gt) == len(ctx_pr)
	cnt = 0

	for idx, (itm1, itm2) in enumerate(zip(ctx_gt, ctx_pr)):
		# print(idx, itm1, itm2)
		str1 = itm1.strip().lower()
		str2 = itm2.strip().lower()


		if (str2 in str1) or (str2[0:-1] in str1) or (str2 in (str1 + " cells")) or (str2 in (str1 + " stem cells")):
			cnt += 1
		# else:			
		# 	print(idx, "=====")
		# 	print(str1, '|', str2)
		# elif (str1 in str2):
		# 	cnt += 1

	return cnt / len(ctx_gt)

# edge1 = (METTL3, mediate, m6A modification)
# edge2 = (METTL3, downregualte, m6A modification)

def score(gtLst, preLst, ty='f1'):
	gt_flat = []
	pred_flat = []

	for (sample_gt, sample_pred) in zip(gtLst, preLst):
		union = set()
		union.update(sample_gt)
		union.update(sample_pred)

		for s in union:
			if s in sample_gt:
				gt_flat.append(1)
				# if isNode:
					
				# else:
				# 	gt_flat.append(1)
					# gt_flat.append(edgeDict[s[1]])
			else:
				gt_flat.append(0)

			if s in sample_pred:
				pred_flat.append(1)

				# if isNode:
				# 	pred_flat.append(1)
				# else:
				# 	pred_flat.append(1)
					# gt_flat.append(edgeDict[s[1]])
			else:
				pred_flat.append(0)

	# print(gt_flat)
	# print(pred_flat)
	# return f1_score(gt_flat, pred_flat, average='macro')
	print('stat', sum(gt_flat), sum(pred_flat))
	if ty == 'f1':
		print('f1')
		return f1_score(gt_flat, pred_flat)
	elif ty == 'recall':
		print('recall')
		return recall_score(gt_flat, pred_flat)
	elif ty == 'precision':
		print('precision')
		return precision_score(gt_flat, pred_flat)


if __name__ == "__main__":
	### load file
	# gtlst = [[each title edge, each edge is a tuple (node1, relationship, node2)], [], []]
	# ctx_gt: context list, length = number of title

	gt_file = '../data/map_final.csv' #'groundtruth.csv'
	# pr_file = '../data/baseline_2_3_edge.csv' 
	# pr_file = '../data/map_baseline_0_1_edge.csv' 
	# pr_file = '../data/map_baseline_1_4_edge.csv' # <<
	pr_file = '../data/map_baseline_2_4_edge.csv' # <<
	# pr_file = '../data/SemRep_edge.csv' # 'prediction.csv'
	# pr_file = 'EIDOs_triples_new.csv' # <<
	# pr_file = 'Reach_triples_new.csv' # <<

	gtLst, ctx_gt, gt_idx = load_file(finalename=gt_file)
	preLst, ctx_pr, pr_idx = load_file(finalename=pr_file)
	# gtLst, ctx_gt, gt_idx, preLst, ctx_pr, pr_idx = load_2file(gt_file, pr_file)

	print(len(gtLst), len(preLst))
	print(len(ctx_gt), len(ctx_pr))
	print(len(gt_idx), len(pr_idx))
	assert len(gtLst) == len(preLst), "title # should be the same"

	for idx1, idx2 in zip(gt_idx, pr_idx):
		if idx1 != idx2:
			print(idx1, idx2)

	ctx_score = getAcc(ctx_gt, ctx_pr)
	# ctx_score = f1_score(ctx_gt, ctx_pr, average='macro')
	print("Context Score", ctx_score)

	gt_node, pre_node, gtLst, preLst = node_normalization(gtLst, preLst)

	# # # calcualte node
	# ttpe = 'f1'
	# ttpe = 'recall'
	ttpe = 'precision'
	nodescore = score(gt_node, pre_node, ty = ttpe)
	print("Node Score", nodescore)

	edgescore = score(gtLst, preLst, ty = ttpe)
	print("Edge score", edgescore)


# Context Score 0.7375
# Node Score 0.9102187397975839
# Edge score 0.44103234237177386
# (base) wuxidong@s-MacBook-Pro-2 Experiment % python evaluation.py
# Context Score 0.875
# Node Score 0.9457714654615881
# Edge score 0.5876902713434812
# (base) wuxidong@s-MacBook-Pro-2 Experiment % python evaluation.py
# Context Score 0.8925
# Node Score 0.9553772070626004
# Edge score 0.639432815665091

