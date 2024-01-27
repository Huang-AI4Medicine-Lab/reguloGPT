import pandas as pd
import gilda
from goatools import obo_parser
import requests
import tempfile

# Cache for storing normalization results
normalization_cache = {}

mapping = {
    "regulate": "AFFECTS",
    "modulate": "AFFECTS",
    "control": "AFFECTS",
    "influence": "AFFECTS",
    "modify": "AFFECTS",
    "contribute": "AFFECTS",
    "affect": "AFFECTS",
    "involve": "AFFECTS",
    "alter": "AFFECTS",
    "promote": "STIMULATES",
    "upregulate": "STIMULATES",
    "activate": "STIMULATES",
    "enhance": "STIMULATES",
    "accelerate": "STIMULATES",
    "amplify": "STIMULATES",
    "induce upregulation of": "STIMULATES",
    "stimulate": "STIMULATES",
    "inhibit": "INHIBITS",
    "suppress": "INHIBITS",
    "attenuate": "INHIBITS",
    "downregulate": "INHIBITS",
    "repress": "INHIBITS",
    "block": "INHIBITS",
    "restrain": "INHIBITS",
    "silence": "INHIBITS",
    "impede": "INHIBITS",
    "prevent": "INHIBITS",
    "reduce": "INHIBITS",
    "decrease": "INHIBITS",
    "mitigate": "INHIBITS",
    "alleviate": "INHIBITS",
    "reduct": "INHIBITS",
    "counteract": "INHIBITS",
    "implicate": "INHIBITS",
    "decrease methylation of": "INHIBITS",
    "mediate downregulation of": "INHIBITS",
    "retard": "INHIBITS",
    "induce": "CAUSES",
    "cause": "CAUSES",
    "lead to": "CAUSES",
    "initiate": "CAUSES",
    "induce activation of": "STIMULATES",
    "induce suppression of": "INHIBITS",
    "derive resistance to": "CAUSES",
    "facilitate": "AUGMENTS",
    "up-regulate": "AUGMENTS",
    "enhance": "AUGMENTS",
    "augment": "AUGMENTS",
    "potentiate": "AUGMENTS",
    "potentiate resistance to": "AUGMENTS",
    "amplify": "AUGMENTS",
    "bind": "INTERACTS_WITH",
    "target": "INTERACTS_WITH",
    "mediate": "INTERACTS_WITH",
    "bind to": "INTERACTS_WITH",
    "associate": "INTERACTS_WITH",
    "maintain": "MAINTAINS",
    "sustain": "MAINTAINS",
    "stabilize": "MAINTAINS",
    "demethylate": "CONVERTS_TO",
    "methylate": "CONVERTS_TO",
    "phosphorylate": "CONVERTS_TO",
    "prevent": "PREVENTS",
    "protect": "PREVENTS",
    "counteract": "PREVENTS",
    "guard": "PREVENTS",
    "aggravate": "COMPLICATES",
    "exacerbate": "COMPLICATES",
    "complicate": "COMPLICATES",
    "treat": "TREATS",
    "ameliorate": "TREATS",
    "remedy": "TREATS",
    "heal": "TREATS",
    "diagnose": "DIAGNOSES",
    "detect": "DIAGNOSES",
    "identify": "DIAGNOSES",
    "produce": "PRODUCES",
    "secrete": "PRODUCES",
    "generate": "PRODUCES",
    "yield": "PRODUCES",
    "emit": "PRODUCES",
    "is in": "LOCATES",
    "locate": "LOCATES",
    "position": "LOCATES",
    "precede": "PRECEDES",
    "antecede": "PRECEDES",
    "lead": "PRECEDES",
    "coexist": "COEXISTS_WITH",
    "accompany": "COEXISTS_WITH",
    "associate": "COEXISTS_WITH",
    "is a": "ISA",
    "constitute": "ISA",
    "represent": "ISA",
    "manifest": "MANIFESTS",
    "express": "MANIFESTS",
    "exhibit": "MANIFESTS",
    "method": "METHODS",
    "approach": "METHODS",
    "technique": "METHODS",
    "occur in": "OCCURS_IN",
    "happen in": "OCCURS_IN",
    "take place in": "OCCURS_IN",
    "part of": "PART_OF",
    "comprise": "PART_OF",
    "consist of": "PART_OF",
    "belong to": "PART_OF",
    "predispose": "PREDISPOSES",
    "sensitize": "PREDISPOSES",
    "prime": "PREDISPOSES",
    "process of": "PROCESS_OF",
    "involve in": "PROCESS_OF",
    "engage in": "PROCESS_OF",
    "compared with": "COMPARED_WITH",
    "liken": "COMPARED_WITH",
    "equate": "COMPARED_WITH",
    "higher than": "HIGHER_THAN",
    "exceed": "HIGHER_THAN",
    "surpass": "HIGHER_THAN",
    "lower than": "LOWER_THAN",
    "under": "LOWER_THAN",
    "beneath": "LOWER_THAN",
    "same as": "SAME_AS",
    "identical to": "SAME_AS",
    "equivalent": "SAME_AS",
    "impair": "INHIBITS",
    "increase": "AUGMENTS",
    "is on": "LOCATES",
    "use": "USES",
    "trigger": "CAUSES",
    "mediate the instability of": "AFFECTS",
    "negatively modulate": "INHIBITS",
    "support": "AUGMENTS",
    "reprogram": "CONVERTS_TO",
    "critical for": "ASSOCIATED_WITH",
    "Exposure to": "ADMINISTERED_TO",
    "regulate the inhibition of": "INHIBITS",
    "evade": "DISRUPTS",
    "exploit": "USES",
    "is required for": "CAUSES",
    "is required": "CAUSES",
    "bridge": "INTERACTS_WITH",
    "degrade": "DISRUPTS",
    "down-regulate": "INHIBITS",
    "drive": "STIMULATES",
    "induce knockdown of": "INHIBITS",
    "positively regulate": "AUGMENTS",
    "mediate the upregulation of":"AUGMENTS",
    "regulate the nuclear shuttling of": "AFFECTS",
    "facilitate": "AUGMENTS",
    "fuels": "CAUSES",
    "fuel": "CAUSES",
    "induce the resistance of": "CAUSES",
    "slow": "INHIBITS",
    "impair autophagy of": "DISRUPTS",
    "induce the abnormal expression of": "DISRUPTS",
    "modulate the stability of": "AFFECTS",
    "influences": "AFFECTS",
    "regulate the stability of": "AFFECTS",
    "contribute to": "CAUSES",
    "induce degradation of": "DISRUPTS",
    "is involved in": "LOCATES",
    "induce loss of": "INHIBITS",
    "destabilize": "DISRUPTS",
    "intensify": "STIMULATES",
    "induce loss of": "INHIBITS",
    "cooperate": "INTERACTS_WITH",
    "regulate scaffold function of": "AFFECTS",
    "compromises": "COMPLICATES",
    "required for": "CAUSES",
    "enhance motility of": "AUGMENTS",
    "confer": "PRODUCES",
    "induce deficiency of": "INHIBITS",
    "catalyses": "AUGMENTS",
    "depend on": "INTERACTS_WITH",
    "elevate": "AUGMENTS",
    "improve": "AUGMENTS",
    "cleave": "DISRUPTS",
    "remit": "INHIBITS",
    "promote growth of": "AUGMENTS",
    "disrupt": "DISRUPTS",
    "essential for": "PREDISPOSES",
    "is essential for": "PREDISPOSES",
    "benefit": "STIMULATES",
    "negatively regulate": "INHIBITS",
    'regulates': 'AFFECTS', 'controls': 'AFFECTS', 'influences': 'AFFECTS', 'modify': 'AFFECTS',
           'contributes': 'AFFECTS', 'affects': 'AFFECTS', 'alters': 'AFFECTS',
           'mediate the instability of': 'AFFECTS', 'regulates the stability of': 'AFFECTS',
           'regulate scaffold function of': 'AFFECTS', 'influences': 'AFFECTS', 'implicates': 'AFFECTS',
           'contributes to': 'AFFECTS', 'mediates': 'AFFECTS', 'mediate': 'AFFECTS', 'mediating': 'AFFECTS', 'mediated':'AFFECTS',
           'modulating': 'AFFECTS', 'modulates': 'AFFECTS', 'modulate': 'AFFECTS','promote': 'STIMULATES', 'promotes': 'STIMULATES', 'promoting': 'STIMULATES',
           'upregulates': 'STIMULATES', 'activates': 'STIMULATES', 'activating': 'STIMULATES', 'activate': 'STIMULATES',
           'activated': 'STIMULATES','stimulates': 'STIMULATES', 'stimulating': 'STIMULATES',
           'accelerates': 'STIMULATES', 'accelerating': 'STIMULATES',  'accelerate': 'STIMULATES',

           'induce upregulation of': 'STIMULATES', 'stimulates': 'STIMULATES', 'drives': 'STIMULATES', 'drive': 'STIMULATES', 'drived': 'STIMULATES', 'driving': 'STIMULATES',
           'intensify': 'STIMULATES', 'intensifies': 'STIMULATES', 'intensified': 'STIMULATES','promote growth of': 'STIMULATES', 'facilitates': 'STIMULATES',
           'up-regulates': 'STIMULATES', 'up-regulating': 'STIMULATES', 'up-regulate': 'STIMULATES','potentiates': 'STIMULATES', 'mediate the upregulation of': 'STIMULATES',
           'positively regulate': 'STIMULATES', 'improve': 'STIMULATES', 'improved': 'STIMULATES','improves': 'STIMULATES', 'improving': 'STIMULATES','support': 'STIMULATES',
           'elevate': 'STIMULATES', 'elevated': 'STIMULATES','elevates': 'STIMULATES', 'elevating': 'STIMULATES','fuels': 'STIMULATES', 'fuel': 'STIMULATES', 'sensitize': 'STIMULATES',
           'enhances': 'AUGMENTS', 'enhancing': 'AUGMENTS','enhance': 'AUGMENTS','amplify': 'AUGMENTS', 'benefits': 'AUGMENTS', 'augments': 'AUGMENTS',
           'increases': 'AUGMENTS', 'increasing': 'AUGMENTS', 'increase': 'AUGMENTS', 'increased': 'AUGMENTS',
           'catalyses': 'AUGMENTS', 'inhibits': 'INHIBITS', 'inhibit': 'INHIBITS','inhibiting': 'INHIBITS','suppresses': 'INHIBITS', 'suppressed': 'INHIBITS', 'suppressing': 'INHIBITS', 'suppress': 'INHIBITS',
           'attenuate': 'INHIBITS', 'attenuates': 'INHIBITS', 'attenuating': 'INHIBITS','downregulate': 'INHIBITS', 'downregulating': 'INHIBITS', 'represses': 'INHIBITS', 'block': 'INHIBITS', 'blocks': 'INHIBITS',
           'restrain': 'INHIBITS', 'silences': 'INHIBITS', 'impedes': 'INHIBITS', 'reduce': 'INHIBITS', 'reduced': 'INHIBITS', 'reducing': 'INHIBITS',
           'reduces': 'INHIBITS','decreases': 'INHIBITS', 'decreasing': 'INHIBITS', 'decrease': 'INHIBITS','mitigates': 'INHIBITS', 'alleviate': 'INHIBITS', 'reduct': 'INHIBITS',
           'decrease methylation of': 'INHIBITS', 'mediate downregulation of': 'INHIBITS', 'retard': 'INHIBITS',
           'induces suppression of': 'INHIBITS', 'negatively modulate': 'INHIBITS', 'down-regulate': 'INHIBITS', 'down-regulating': 'INHIBITS',
           'induce knockdown of': 'INHIBITS', 'negatively regulate': 'INHIBITS', 'slows': 'INHIBITS',
           'induce loss of': 'INHIBITS', 'remit': 'INHIBITS', 'degrade': 'INHIBITS', 'degrading': 'INHIBITS',
           'induce deficiency of': 'INHIBITS', 'decay': 'INHIBITS', 'disrupt': 'DISRUPTS',"impair": "INHIBITS", "impairs": "INHIBITS",
           'impair autophagy of': 'DISRUPTS', 'induce the abnormal expression of': 'DISRUPTS',
           'modulate the stability of': 'DISRUPTS', 'destabilize': 'DISRUPTS', 'cleave': 'DISRUPTS','destabilizes': 'DISRUPTS', 'destabilized': 'DISRUPTS', 'destabilizing': 'DISRUPTS',
           'prevent': 'PREVENTS', 'prevents': 'PREVENTS','preventing': 'PREVENTS','prevented': 'PREVENTS','protect': 'PREVENTS', 'counteract': 'PREVENTS', 'guard': 'PREVENTS',
           'induce the resistance of': 'PREVENTS', 'derive resistance to': 'PREVENTS',
           'potentiate resistance to': 'PREVENTS', 'evade': 'PREVENTS', 'promote resistance to': 'PREVENTS',
           'induce': 'CAUSES', 'induces': 'CAUSES', 'induced': 'CAUSES', 'inducing': 'CAUSES', 'cause': 'CAUSES', 'lead': 'CAUSES', 'leads': 'CAUSES',
           'lead to': 'CAUSES', 'result in': 'CAUSES', 'initiate': 'CAUSES',
           'trigger': 'CAUSES', 'triggers': 'CAUSES', 'triggering': 'CAUSES','is required for': 'CAUSES', 'is required': 'CAUSES', 'required for': 'CAUSES',
           'bind': 'INTERACTS_WITH', 'target': 'INTERACTS_WITH', 'bind to': 'INTERACTS_WITH',
           'depend on': 'INTERACTS_WITH', 'cooperate': 'INTERACTS_WITH', 'associate': 'ASSOCIATED_WITH',
           'bridge': 'ASSOCIATED_WITH', 'involve': 'ASSOCIATED_WITH', 'maintain': 'MAINTAINS',
           'sustain': 'MAINTAINS', 'stabilize': 'MAINTAINS', 'demethylate': 'CONVERTS_TO',
           'methylate': 'CONVERTS_TO', 'phosphorylate': 'CONVERTS_TO', 'reprogram': 'CONVERTS_TO',
           'aggravate': 'COMPLICATES', 'exacerbate': 'COMPLICATES', 'complicate': 'COMPLICATES',
           'compromise': 'COMPLICATES', 'treat': 'TREATS', 'ameliorate': 'TREATS', 'remedy': 'TREATS',
           'heal': 'TREATS', 'diagnose': 'DIAGNOSES', 'detect': 'DIAGNOSES', 'identify': 'DIAGNOSES',
           'recognize': 'DIAGNOSES', 'secrete': 'PRODUCES', 'produce': 'PRODUCES', 'generate': 'PRODUCES',
           'yield': 'PRODUCES', 'emit': 'PRODUCES', 'confer': 'PRODUCES', 'is in': 'LOCATES', 'locate': 'LOCATES',
           'position': 'LOCATES', 'is on': 'LOCATES', 'occur in': 'OCCURS_IN', 'happen in': 'OCCURS_IN',
           'take place in': 'OCCURS_IN', 'precede': 'PRECEDES', 'antecede': 'PRECEDES', 'coexist': 'COEXISTS_WITH',
           'accompany': 'COEXISTS_WITH', 'manifest': 'MANIFESTS', 'express': 'MANIFESTS', 'exhibit': 'MANIFESTS',
           'method': 'METHODS', 'approach': 'METHODS', 'technique': 'METHODS', 'is a': 'ISA', 'represent': 'ISA',
           'part of': 'PART_OF', 'comprise': 'PART_OF', 'consist of': 'PART_OF', 'belong to': 'PART_OF',
           'constitute': 'PART_OF', 'predispose': 'PREDISPOSES', 'process of': 'PROCESS_OF',
           'involve in': 'PROCESS_OF', 'is involved in': 'PROCESS_OF', 'engage in': 'PROCESS_OF',
           'compared with': 'COMPARED_WITH', 'liken': 'COMPARED_WITH', 'equate': 'COMPARED_WITH',
           'higher than': 'HIGHER_THAN', 'exceed': 'HIGHER_THAN', 'surpass': 'HIGHER_THAN',
           'lower than': 'LOWER_THAN', 'under': 'LOWER_THAN', 'beneath': 'LOWER_THAN', 'same as': 'SAME_AS',
           'identical to': 'SAME_AS', 'equivalent': 'SAME_AS', 'use': 'USES', 'exploit': 'USES',
           'exposure to': 'ADMINISTERED_TO', 'regulates': 'AFFECTS',
}

# Function to download and save the GO OBO file
def download_go_obo_file(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, mode='wb', suffix='.obo') as temp_file:
        temp_file.write(response.content)
        return temp_file.name


# Function to load GO terms
def load_go_terms(file_path):
    return obo_parser.GODag(file_path)


# Function to search GO terms for a specific search term
def search_go_term(go_terms, search_term):
    for go_id, go_term in go_terms.items():
        if search_term.lower() in go_term.name.lower():
            return f"{search_term.lower()}"
    return None


# Download and load GO terms
go_obo_file_path = download_go_obo_file('http://purl.obolibrary.org/obo/go/go-basic.obo')
go_terms = load_go_terms(go_obo_file_path)

def normalize_edge(row, column):
    edge = row[column]
    # df['edge'] = df['edge'].str.lower().map(mapping).fillna(df['edge'])

    if pd.notna(edge) and edge.strip():
        return mapping.get(edge.lower().strip(), edge)  # Returns the normalized edge, or the original edge if not found in the map
    return edge
# Modified normalize_term function with cache checking
def normalize_term(term, context):
    term = str(term).strip()

    # Check cache first
    if term in normalization_cache:
        return normalization_cache[term]

    scored_matches = gilda.ground(term, context=context)
    if scored_matches:
        if term.lower() in ["m6a", "m6A", "M6A"]:
            # normalization_result = f"{scored_matches[2].term.id}"
            normalization_result = "m6A"
        else:
            # match = scored_matches[0]
            # normalization_result = f"{match.term.db}:{match.term.id}"
            normalization_result = f"{scored_matches[0].term.entry_name}"

    else:
        # Fallback to GO terms if Gilda fails
        normalization_result = search_go_term(go_terms, term) or term

    # Store result in cache
    # if (normalization_result == "MESH:C010223" or normalization_result == "N(6)-Methyladenosine (m6A)"
    #         or "N(6)-Methyladenosine" in normalization_result or "N6-Methyladenosine" in normalization_result):
    #     normalization_result = "CHEBI:21891"
    normalization_cache[term] = normalization_result
    return normalization_result


# Load your CSV data
csv_file_path = 'test1.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file_path)


def normalize_if_not_empty(row, column, context):
    term = row[column]
    if pd.notna(term) and term.strip():
        return normalize_term(term, context)
    return term


data['normalized_node_A'] = data.apply(lambda row: normalize_if_not_empty(row, 'node-A', row['Title']), axis=1)
data['normalized_node_B'] = data.apply(lambda row: normalize_if_not_empty(row, 'node-B', row['Title']), axis=1)
data['normalized_edge'] = data.apply(lambda row: normalize_edge(row, 'edge'), axis=1)
data['normalized_Context'] = data.apply(lambda row: normalize_if_not_empty(row, 'Context', row['Title']), axis=1)

# Save the results back to a new CSV file
output_file_path = 'test2.csv'  # You can choose a different file name
data.to_csv(output_file_path, index=False)

# Display the first few rows of the DataFrame to see the results
print(data.head())
