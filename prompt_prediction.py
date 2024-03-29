import openai
import os

import pandas as pd

import time
import csv

openai.api_key = 'API_KEY'#



def load(filnename="test.tsv"):
	
	# open .tsv file
	lst = []
	with open(filnename) as file: 
		# Passing the TSV file to reader() function with tab delimiter 
		# This function will read data from file
		# tsv_file = csv.reader(file, delimiter="\t")
		tsv_file = csv.reader(file, delimiter=",")

		# printing data line by line
		for idx, line in enumerate(tsv_file):
			if idx == 0:
				continue
			# print(line, len(line))
			lst.append([line[0], line[1]])
			# print(idx, '==>', line[1])

	return lst

# https://www.pythontutorial.net/python-basics/python-write-csv-file/
def write(data=None, fieldnames=None, filnename="title.csv"):
	with open(filnename, 'w', encoding='UTF8', newline='') as f:
	    writer = csv.writer(f)

	    # write the header
	    writer.writerow(fieldnames)
	    # write the data
	    for item in data:
	    	writer.writerow(item)




# model="gpt-3.5-turbo"
# def get_completion(prompt, model="gpt-4-1106-preview"):
def get_completion(prompt, model="gpt-4"):

	messages = [{"role": "system", "content": prompt}]
	ignore = 0
	while True:
		try:
			response = openai.ChatCompletion.create(
				model=model,
				messages=messages,
				temperature=0,
			)
			time.sleep(0.5)
			break

		except Exception as e:
			print(">>>>>>>>>>>>>>>>>>>>")
			print(e)
			if ("limit" in str(e)):
				time.sleep(2)
			else:
				ignore += 1
				print('ignored', ignore)
				assert 2 ==3, "Error occurs, Check!"
				break

	return response.choices[0].message["content"]

#No node. Methylation 


prompt_dict = {
	"baseline": "Construct a graph including connected triplets (head node, edge, tail node) to represent the molecular regulatory relationship described in the provided sentence. Follow these explicit guidelines:\
1. Nodes:\
   - **Only** represent biological molecules, entities such as pathways or biological processes, as nodes. Denote a node as a single biology entity.\
   - **Do Not** include context in the node unless the context is a complete node. For example, use only differentiation in 'mouse embryonic stem cell differentiation' as the node since the 'mouse embryonic stem cell' is the context. But node must be a noun biological molecules or entities.\
   - Regard phrases like the A/B/C or (A-B-C) “axis”, “pathway”, and “signaling” as a node. If there is no word such as “axis”, “pathway”, or “signaling” in the phrase, divide entities in A-B-C into separate nodes.  \
   - **Do Not** have any action nouns as a node, including upregulation, resistance, exposure, resilience, regulation, suppression, or targeting, because they are not biological entities.\
   - **Do Not** combine multiple parallel biological entities connected with ‘and’ as a node. Extract each entity as a separate node.\
   - Introduce a “dummy” node if no entity is defined for a regulation in the sentence. \
2. Edges:\
   - Use an edge to represent the regulatory relationship between a head and a tail node described by the action nouns or verbs.\
   - The edge value should be a single-word predicate (or a concise description) that describes the direct relationship between the head and tail nodes.\
   - Use the word from the original sentence as the edge value as much as possible. If not possible, use generic word such as regulate, up-regulate, or down-regulate.\
   - 'is on' can be used as a relationship in the edge. **Do Not** use other prepositions, such as 'of', or 'in' as a relationship.\
   - **Do Not** include more than two nodes in an edge. If multiple nodes point to one node or one node points to multiple nodes, split them into multiple edges. \
3. Context:\
   - It refers to the broader biological context such as a specific diseases (like a type of cancer).\
   - Directly use phrases in the input.\
   - **Do not** include biological processes as a part of the context. Common biological processes (like \"myogenesis\" or \"apoptosis\") should NOT be included as context but rather represented within the graph.\
   - ***Do not** just include unclear or general biological entities, such as 'tumor', 'yeast'. In this case, indicate it as 'Null'.\
   - If there's no context mentioned, indicate it as 'Null'.\
4. Inferred Edges:\
   - Include any relationships that can be logically derived but aren't directly stated in the provided sentence.\
   - **Only** include inferred edge to make the graph complete. Do not introduce extra relationships. \
For each sentence, format your answer as:\
- Context => [Context or 'Null']\
- Graph:\
  - nodes=> a list of nodes. For example, Node 1: HNRNPA2B1, Node 2: Progression.\
  - Direct edges=> a list of direct edges with descriptions. For example, Edge 1: From Node 1 to Node 2; edge value: promote. It denotes that HNRNPA2B promotes progression.\
  - Inferred edges=> a list of inferred edges with descriptions\
Sentence:",

	"fewshot": "Construct a graph including connected triplets (head node, edge, tail node) to represent the molecular regulatory relationship described in the provided sentence. Follow these explicit guidelines:\
1. Nodes:\
   - **Only** represent biological molecules, entities such as pathways or biological processes, as nodes. Denote a node as a single biology entity.\
   - **Do Not** include context in the node unless the context is a complete node. For example, use only differentiation in 'mouse embryonic stem cell differentiation' as the node since the 'mouse embryonic stem cell' is the context. But node must be a noun biological molecules or entities.\
   - Regard phrases like the A/B/C or (A-B-C) “axis”, “pathway”, and “signaling” as a node. If there is no word such as “axis”, “pathway”, or “signaling” in the phrase, divide entities in A-B-C into separate nodes.  \
   - **Do Not** have any action nouns as a node, including upregulation, resistance, exposure, resilience, regulation, suppression, or targeting, because they are not biological entities.\
   - **Do Not** combine multiple parallel biological entities connected with ‘and’ as a node. Extract each entity as a separate node.\
   - Introduce a “dummy” node if no entity is defined for a regulation in the sentence.\
2. Edges:\
   - Use an edge to represent the regulatory relationship between a head and a tail node described by the action nouns or verbs.\
   - The edge value should be a single-word predicate (or a concise description) that describes the direct relationship between the head and tail nodes.\
   - Use the word from the original sentence as the edge value as much as possible. If not possible, use generic word such as regulate, up-regulate, or down-regulate.\
   - 'is on' can be used as a relationship in the edge. **Do Not** use other prepositions, such as 'of', or 'in' as a relationship.\
   - **Do Not** include more than two nodes in an edge. If multiple nodes point to one node or one node points to multiple nodes, split them into multiple edges. \
3. Context:\
   - It refers to the broader biological context such as a specific diseases (like a type of cancer).\
   - Directly use phrases in the input.\
   - **Do not** include biological processes as a part of the context. Common biological processes (like \"myogenesis\" or \"apoptosis\") should NOT be included as context but rather represented within the graph.\
   - ***Do not** just include unclear or general biological entities, such as 'tumor', 'yeast'. In this case, indicate it as 'Null'.\
   - If there's no context mentioned, indicate it as 'Null'.\
4. Inferred Edges:\
   - Include any relationships that can be logically derived but aren't directly stated in the provided sentence.\
   - **Only** include inferred edge to make the graph complete. Do not introduce extra relationships. \
For each sentence, format your answer as:\
- Context => [Context or 'Null']\
- Graph:\
  - nodes=> [List of nodes]\
  - Direct edges=> [List of direct edges with descriptions]\
  - Inferred edges=> [List of inferred edges with descriptions]\
Sentence 1: GR-mediated FTO transactivation induces lipid accumulation in hepatocytes via demethylation of m6A on lipogenic mRNAs. \
Answer:\
Context => Hepatocytes\
Graph:\
- nodes=> Node 1: GR, Node 2: FTO transactivation, Node 3: Lipid Accumulation, Node 4: m6A, Node 5: lipogenic mRNAs.\
- Direct edges=> Edge 1: From Node 1 to Node 2; edge value: mediate. It represents that GR mediates FTO transactivation. Edge 2: From Node 2 to Node 3; edge value: induce. It represents that FTO transactivation induces lipid accumulation. Edge 3: From Node 2 to Node 4; edge value: demethylate. It represents that FTO demethylates m6A. Edge 4: From Node 4 to Node 5; edge value: is on. It represents that m6A is on lipogenic mRNAs. \
- Inferred edges=> Edge 5: From Node 5 to Node 3; edge value: regulate. It implies that demethylation of m6A on lipogenic mRNAs regulates lipid accumulation.\
Sentence2: m6A methyltransferase METTL3 facilitates tumor progression via an m6A-IGF2BP2-dependent mechanism in colorectal carcinoma.\
Answer:\
Context => Colorectal Carcinoma\
Graph:\
- nodes=> Node 1: METTL3, Node 2: Tumor Progression, Node 3: m6A, Node 4: IGF2BP2.\
- Direct edges=> Edge 1: From Node 1 to Node 2; edge value: facilitate. It represents that METTL3 facilitates tumor progression. Edge 2: From Node 1 to Node 3; edge value: regulate. It represents that METTL3 regulates m6A. Edge 3: From Node 4 to Node 3; edge value: bind. It represents that IGF2BP2 binds m6A. \
- Inferred edges=> Edge 4: From Node 3 to Node 2; edge value: facilitate. It implies that m6A facilitate Tumor Progression.\
Sentence3: Loss of m6A on FAM134B promotes IL-1beta-induced adipogenesis in porcine adipocytes through m6A-YTHDF2-dependent way.\
Answer:\
Context => Porcine Adipocytes\
Graph:\
- nodes=> Node 1: dummy, Node 2: m6A, Node 3: FAM134B, Node 4: IL-1beta, Node 5: adipogenesis, Node 6: YTHDF2.\
- Direct edges=> Edge 1: From Node 1 to Node 2; edge value: induce loss of. It represents the loss of m6A. Edge 2: From Node 2 to Node 3; edge value: is on. It represents m6A is on FAM134B. Edge 3: From Node 3 to Node 5; edge value: promote. It represents that FAM134B promotes adipogenesis. Edge 4: From Node 4 to Node 5; edge value: induce. It represents that IL-1beta induces adipogenesis. Edge 5: From Node 6 to Node 2; edge value: bind. It represents that m6a reader YTHDF2 binds m6a.  \
- Inferred edges=> Null.\
Sentence4: The 18S rRNA m6A methyltransferase METTL5 promotes mouse embryonic stem cell differentiation.\
Answer:\
Context => Mouse embryonic stem cell\
Graph:\
- nodes=> Node 1: METTL5, Node 2: differentiation.\
- Direct edges=> Edge 1:  From Node 1 to Node 2; edge value: promote. It represents that METTL5 promotes mouse embryonic stem cell differentiation. \
- Inferred edges=> Null.\
Sentence:",

	"cot" : "Construct a graph including connected triplets (head node, edge, tail node) to represent the molecular regulatory relationship described in the provided sentence. Follow these explicit guidelines:\
1. Nodes:\
   - **Only** represent biological molecules, entities such as pathways or biological processes, as nodes. Denote a node as a single biology entity.\
   - **Do Not** include context in the node unless the context is a complete node. For example, use only differentiation in 'mouse embryonic stem cell differentiation' as the node since the 'mouse embryonic stem cell' is the context. But node must be a noun biological molecules or entities.\
   - Regard phrases like the A/B/C or (A-B-C) “axis”, “pathway”, and “signaling” as a node. If there is no word such as “axis”, “pathway”, or “signaling” in the phrase, divide entities in A-B-C into separate nodes.  \
   - **Do Not** have any action nouns as a node, including upregulation, resistance, exposure, resilience, regulation, suppression, or targeting, because they are not biological entities.\
   - **Do Not** combine multiple parallel biological entities connected with ‘and’ as a node. Extract each entity as a separate node.\
   - Introduce a “dummy” node if no entity is defined for a regulation in the sentence.\
2. Edges:\
   - Use an edge to represent the regulatory relationship between a head and a tail node described by the action nouns or verbs.\
   - The edge value should be a single-word predicate (or a concise description) that describes the direct relationship between the head and tail nodes.\
   - Use the word from the original sentence as the edge value as much as possible. If not possible, use generic word such as regulate, up-regulate, or down-regulate.\
   - 'is on' can be used as a relationship in the edge. **Do Not** use other prepositions, such as 'of', or 'in' as a relationship.\
   - **Do Not** include more than two nodes in an edge. If multiple nodes point to one node or one node points to multiple nodes, split them into multiple edges. \
3. Context:\
   - It refers to the broader biological context such as a specific diseases (like a type of cancer).\
   - Directly use phrases in the input.\
   - **Do not** include biological processes as a part of the context. Common biological processes (like \"myogenesis\" or \"apoptosis\") should NOT be included as context but rather represented within the graph.\
   - ***Do not** just include unclear or general biological entities, such as 'tumor', 'yeast'. In this case, indicate it as 'Null'.\
   - If there's no context mentioned, indicate it as 'Null'.\
4. Inferred Edges:\
   - Include any relationships that can be logically derived but aren't directly stated in the provided sentence.\
   - **Only** include inferred edge to make the graph complete. Do not introduce extra relationships. \
For each sentence, format your answer as:\
- Context => [Context or 'Null']\
- Graph:\
  - nodes=> [List of nodes]\
  - Direct edges=> [List of direct edges with descriptions]\
  - Inferred edges=> [List of inferred edges with descriptions]\
Sentence 1: GR-mediated FTO transactivation induces lipid accumulation in hepatocytes via demethylation of m6A on lipogenic mRNAs. \
Answer: This regulation happens in Hepatocytes so the context is Hepatocytes. The sentence has five biological entities: GR, FTO transactivation, Lipid Accumulation, m6A, and lipogenic mRNAs. 'of' is not a relationship and demethylation is an action noun so we use m6A as a node. ‘GR-mediated FTO transactivation’ means that there is a direct edge between GR and FTO transactivation. There is also a direct edge from FTO transactivation to lipid accumulation because FTO transactivation is the subject and lipid accumulation is the object of the sentence. The phrase after 'via' defines the mechanism, by which FTO transactivation induces lipid accumulation. It indicates that FTO transactivation demethylates m6A on lipogenic mRNAs to which induces lipid accumulation. As a result, we have direct edges, one from FTO transactivation to m6A and the other from m6A to lipogenic mRNAs. Finally, we have inferred edges from lipogenic mRNAs to lipid accumulation to describe the outcome of the mechanism and complete the graph.\n\
Context => Hepatocytes\
Graph:\
- nodes=> Node 1: GR, Node 2: FTO transactivation, Node 3: Lipid Accumulation, Node 4: m6A, Node 5: lipogenic mRNAs.\
- Direct edges=> Edge 1: From Node 1 to Node 2; edge value: mediate. It represents that GR mediates FTO transactivation. Edge 2: From Node 2 to Node 3; edge value: induce. It represents that FTO transactivation induces lipid accumulation. Edge 3: From Node 2 to Node 4; edge value: demethylate. It represents that FTO demethylates m6A. Edge 4: From Node 4 to Node 5; edge value: is on. It represents that m6A is on lipogenic mRNAs. \
- Inferred edges=> Edge 5: From Node 5 to Node 3; edge value: regulate. It implies that demethylation of m6A on lipogenic mRNAs regulates lipid accumulation.\
Sentence2: m6A methyltransferase METTL3 facilitates tumor progression via an m6A-IGF2BP2-dependent mechanism in colorectal carcinoma.\
Answer: This regulation happens in Colorectal Carcinoma, so the context is Colorectal Carcinoma. The sentence has four biological entities: METTL3, Tumor Progression, m6A, and IGF2BP2. m6A methyltransferase is the appositive of METTL3 and defines that METTL3 is an m6A methylase. Therefore, m6A methyltransferase is **not** a node. There is a direct edge from METTL3 to tumor progression because METTL3 is the subject and tumor progression is the object of the sentence. The phrase after 'via' defines the mechanism, by which METTL3 facilitates tumor progression. because there is no pathway or signaling following 'm6A-IGF2BP2-dependent' so we further divide 'm6A-IGF2BP2-dependent' into separate nodes. Therefore, the first direct edge is from METTL3 to m6A, suggesting that METTL3 regulates m6A. Next, since IGF2BP2 is an m6A reader that binds to m6A, then there is another edge from IGF2BP2 to m6A with edge value “bind”. Finally, we have an inferred edge from m6A to tumor progression to describe the outcome of the mechanism and complete the graph. .\n\
Context => Colorectal Carcinoma\
Graph:\
- nodes=> Node 1: METTL3, Node 2: Tumor Progression, Node 3: m6A, Node 4: IGF2BP2.\
- Direct edges=> Edge 1: From Node 1 to Node 2; edge value: facilitate. It represents that METTL3 facilitates tumor progression. Edge 2: From Node 1 to Node 3; edge value: regulate. It represents that METTL3 regulates m6A. Edge 3: From Node 4 to Node 3; edge value: bind. It represents that IGF2BP2 binds m6A. \
- Inferred edges=> Edge 4: From Node 3 to Node 2; edge value: facilitate. It implies that m6A facilitate Tumor Progression.\
Sentence3: Loss of m6A on FAM134B promotes IL-1beta-induced adipogenesis in porcine adipocytes through m6A-YTHDF2-dependent way.\
Answer: This regulation happens in Porcine Adipocytes, so the context is Porcine Adipocytes. The sentence has six nodes: dummy, m6A, FAM134B, IL-1beta, adipogenesis, and YTHDF2. The reason to have a dummy node is due to 'Loss of m6A’, where ‘loss of’ suggests an edge with edge value “induce loss of’ from something undefined, thus a dummy entity, to m6A. Then, ‘m6A on FAM134B’ suggests a direct edge between m6A and FAM134B with values ‘is on’. Next, there is a direct edge from FAM134B to adipogenesis because m6A on FAM134B promotes adipogenesis. In addition, IL-1beta-induced adipogenesis suggests an edge from IL-1beta to adipogenesis with the edge value “induce”. The phrase after 'through' defines the mechanism, by which m6A on FAM134B promotes adipogenesis. Since there is no pathway or signaling following 'm6A-YTHDF2-dependent', we divide 'm6A-YTHDF2-dependent' into separate nodes “m6A” and “YTHDF2”. Since FAM134B is not an m6A writer/eraser, it can’t regulate m6A. Therefore, this m6A is the m6A on FAM134B. Because of this, we don’t have a separate node for this m6A and we don’t need inferred edges from this m6A to adipogenesis. However, there is an edge from YTHDF2 to m6a because YTHDF2 is a m6a reader and YTHDF2 binds m6a.  \n\
Context => Porcine Adipocytes\
Graph:\
- nodes=> Node 1: dummy, Node 2: m6A, Node 3: FAM134B, Node 4: IL-1beta, Node 5: adipogenesis, Node 6: YTHDF2.\
- Direct edges=> Edge 1: From Node 1 to Node 2; edge value: induce loss of. It represents the loss of m6A. Edge 2: From Node 2 to Node 3; edge value: is on. It represents m6A is on FAM134B. Edge 3: From Node 3 to Node 5; edge value: promote. It represents that FAM134B promotes adipogenesis. Edge 4: From Node 4 to Node 5; edge value: induce. It represents that IL-1beta induces adipogenesis. Edge 5: From Node 6 to Node 2; edge value: bind. It represents that m6a reader YTHDF2 binds m6a.  \
- Inferred edges=> Null.\
Sentence4: The 18S rRNA m6A methyltransferase METTL5 promotes mouse embryonic stem cell differentiation.\
Answer: This regulation happens in mouse embryonic stem cells, which are specific cells and there is not diseases, so the context is mouse embryonic stem cells. Do not include differentiation since it is a process. The sentence has two biological entities: METTL5 and differentiation. The mouse embryonic stem cell is the context, so it is not included in node differentiation. Since 'The 18S rRNA m6A methyltransferase defines METTL5 and is appositive of METTL5 so the node is METTL5. For edges, there is a direct edge from METTL5 to differentiation because METTL5 is the subject and differentiation is the object of the sentence. There are no inferred edges because the graph has been completed.\n\
Context => Mouse embryonic stem cell\
Graph:\
- nodes=> Node 1: METTL5, Node 2: differentiation.\
- Direct edges=> Edge 1:  From Node 1 to Node 2; edge value: promote. It represents that METTL5 promotes mouse embryonic stem cell differentiation. \
- Inferred edges=> Null.\
Sentence:"
}

def generatePrompt(ipt, ptype = 'baseline'):

	prefix = prompt_dict[ptype]
	return prefix + ipt + "Answer:"


def nodes2Dic(ipt):
	nodeLst = ipt.replace(".","").replace("\n","").split(',')
	res = {}
	for idx, item in enumerate(nodeLst):
		item = item.strip()
		entry = item.split(':')
		res[entry[0].strip()] = entry[1].strip()
	return res


def delExplanation(ipt):
	lst = ipt.split('.')
	reS = []
	for idx, item in enumerate(lst):
		if "It" not in item and len(item) > 0:
			reS.append(item + ".")
	reS = ' '.join(reS)

	return reS


if __name__ == "__main__":

	######## Start 
	# tList = load("raw.tsv")
	tList = load("final_pr.csv")
	# ipt = "METTL3 restrains papillary thyroid cancer progression via m6A/c-Rel/IL-8 mediated neutrophils infiltration."
	# tList = [[0, ipt]]
	# print(tList)
	# fieldnames = ['Title', 'gpt4', 'Context', 'nodes', 'Direct edges', 'Inferred edges', 'Direct edges (+)', 'Inferred edges(+)']
	fieldnames = ['Idx',	'Title',	'gpt4']


	for idx, title in enumerate(tList):
		print("processing title ", idx)
		ipt = title[1]
		# prompt = generatePrompt(ipt, ptype = 'cot')
		# prompt = generatePrompt(ipt, ptype = 'fewshot')
		prompt = generatePrompt(ipt, ptype = 'baseline')
		res = get_completion(prompt)
		# print(res)
		tList[idx].append(res)

		write(data=tList, fieldnames=fieldnames, filnename='baseline_0_3.csv')


	###########

# Once you finish constructing the graph, try to generate a sentence based on the graph without considering the original sentence. Compare the generated and original sentences. If they are semantically different, regenerate the graph. Otherwise, output the graph based on the format: - Context -> [Context or 'Null'] - Graph without context: - nodes-> [List of nodes] - Direct edges-> [List of direct edges with descriptions] - Inferred edges-> [List of inferred edges with descriptions]
