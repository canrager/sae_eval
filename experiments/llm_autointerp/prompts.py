import json
import os

### SYSTEM PROMPT ###

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. Your task is to analyze the neuron and score how strong its behavior is related to a concept in {concepts} on an integer scale from {min_scale} to {max_scale} in json format.

Task description:
You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear between delimiters like <<this>>. The activation value of the token is given after each token in parentheses like <<this>>(3). 
You will also be shown a list called promoted tokens. The logits promoted by the neuron shed light on how the neuron's activation influences the model's predictions or outputs. It is possible that this list is more informative than the list of text examples.
For each concept, try to judge whether the neurons behavior is related to the concept.
If part of the text examples or predicited tokens are incorrectly formatted, please ignore them.
If you are not able to find any coherent description of the neurons behavior, decide that the neuron is not related to any concept.

Scoring rubric:
Score 4: The majority of examples, activation scores, and promoted tokens are clearly related to the concept.
Score 3: About half of the examples and promoted tokens are directly related to the concept. 
Score 2: Only some of the examples are directly related to the concept, and some more are distantly related.
Score 1: NONE of the examples is directly related to the concept, but single tokens can be distantly related to the general domain of the concept.
Score 0: NONE of the text examples can be distantly related in any way to the broader field of the concept.

Structure your response as follows:
Step 1. Give a single sentence summary for the full text examples.
Step 2. Give a separate single sentence summary for the promoted tokens.
Step 3. Discuss your decision in 1-3 sentences.
After finishing all steps above, provide a single json block at the end of your response. The json block should contain your scores on an integer scale from {min_scale} to {max_scale} for each concept as shown in the examples.
"""

# Snippets from few shot examples:
"""
"32": {
  "sae": "../dictionary_learning/dictionaries/autointerp_test_data/pythia70m_sweep_topk_ctx128_0730/resid_post_layer_3/trainer_18",
  "class_index": "attorney",
  "sampled_index": 1,
  "sae_feat_index": 6056,
  "example_prompts": [
    "\n\n\nExample 1: Italian senators voted on Wednesday to << lift>>(9) << immunity>>(8) for far- <<right>>(7) leader Matteo Salvini, << opening>>(8) the way for a potentially career-ending trial over << accusations>>(7) he illegally detained migrants at sea last year.\n\nAdvertising Read more\n\nThe decision gives magistrates in Sicily the go-ahead to press charges over his decision to keep 131 rescued migrants blocked aboard a coastguard ship for six days last July as he waited for other European Union states to agree to take them in.\n\n\n\nSalvini, the head of Italy's League party who was serving as interior minister at the time, could eventually face up to 15 years\n\n\n\n\nExample 2: This is an archived article and the information in the article may be outdated. Please look at the time stamp on the story to see when it was last updated.\n\nWASHINGTON \u2014 A federal << judge>>(8) has << ordered>>(8) a temporary << injunction>>(9) << against>>(8) the << California>>(8) law requiring presidential candidates to release their tax returns to secure a spot on the state\u2019s presidential primary ballot \u2014 a law aimed at President Donald Trump, who has not released his tax returns.\n\nIn a ruling Thursday, US District Court Judge Morrison England, Jr., said that California cannot force candidates to disclose their tax returns as outlined in a new state law. England said he would make his\n\n\n\n\nExample 3: that Trump shared highly-classified intelligence with Russian officials; and reports that << Trump>>(8) asked Comey to << drop>>(8) the FBI investigation into former national security << adviser>>(8) Michael Flynn back in Feburary.\n\nRecently, it emerged that << Trump>>(8) reportedly asked top intelligence officials to publicly push << back>>(9) against the Russia probe, and that senior adviser and son-in-law Jared Kushner met with Russian ambassador Sergey Kislyak to discuss the creation of a secret \"backchannel\" between the US and Russia through Russian facilities in an attempt to bypass US surveillance.\n\nThe proposed war room, Axios reported, will be filled with \"experienced veterans from the\n\n\n\n\nExample 4: also being deployed in the area, media reports recently said.\n\nMeanwhile, Israeli Defence Minister Avigdor Lieberman voiced relief at the situation, saying the Syrian << regime>>(7) has restored power in the country. Last week, the Syrian regime forces raised the flag in Quneitra in Golan.\n\nThis came as Israel << launched>>(7) an airst <<rike>>(9) earlier this week, killing Islamic State (IS) << militants>>(7) who reportedly tried << to>>(7) cross inside its controlled part of the Golan Heights.\n\nMainly aiming at keeping Iranian backers off Golan and countering Iran\u2019s influence from growing in Syria, Israel has been on high alert\n\n\n\n\nExample 5: US House passes overhaul of US music laws\n\nWashington (AFP) \u2013 The US House of Representatives << voted>>(7) << unanimously>>(8) << Wednesday>>(8) << to>>(9) << overhaul>>(9) how musicians are compensated for the playing of songs, agreeing to revamp antiquated regulations as streaming shakes up the industry.\n\nThe reform package still needs to go through the Senate, but it enjoyed support across the political spectrum and from industry and artist groups \u2014 a rare consensus in bitterly divided Washington.\n\nThe Music Modernization Act would notably extend copyright protections for songs from before 1972, the cutoff under current law that has set off an avalanche of lawsuits from older artists upset about non-payment.\n\n\n\n\n\n\nExample 6: Colombian rebels free US veteran\n\nAP, BOGOTA\n\nColombia\u2019s main leftist rebel group on Sunday released a former US Army private who the guerrillas seized << in>>(7) June after he refused << to>>(7) heed local << officials>>(8)\u2019 << warnings>>(9) and wandered into << rebel>>(7)-held territory.\n\nKevin Scott Sutay, who is in his late 20s, was quietly turned over to Cuban and Norwegian officials, and the International Committee of the Red Cross in the southeastern region from which he disappeared four months earlier.\n\nUS Secretary of State John Kerry immediately thanked Bogota in a statement for its \u201ctireless efforts\u201d\n\n\n\n\nExample 7: said a \"terrorist group\" fired two rocket shells from al-Lairamoun area on Aleppo University, while activists said a << government>>(8) << troop>>(8) airst <<rike>>(9) << caused>>(7) the carnage.\n\nIt reported that << President>>(8) Bashar al-Assad has given directives to rehabilitate, as soon as possible, what has been destroyed in Aleppo University at the hands of the \"terrorist killers,\" a term the government uses to refer to the rebels.\n\nSyrian Minister of Higher Education Mohammad Yahya Mu'ala was quoted by SANA as saying that the president gave his instructions to rehabilitate Aleppo University to\n\n\n\n\nExample 8: A top adviser to Hillary Clinton\u2019s campaign-in-waiting accused the George W. << Bush>>(8) << administration>>(8) of using private emails to skirt transparency rules in 2007.\n\nJohn Podesta, who left the White House in February for an unofficial role with Clinton, criticized << Bush>>(9) << administration>>(8) << officials>>(8) for using Republican National Committee email accounts for official business.\n\n\u201cAt the end of the day, it looks like they were trying to avoid the Records Act... by operating official business off the official systems,\u201d Podestasaid in an interviewwith The Wall Street Journal.\n\nThe Bush White House admitted that it lost thousands of emails that weren\n\n\n\n\nExample 9: of the Association of California School Administrators, says the state is only shifting its problems onto school districts. The Legislature in February gave authority to the three officers to delay payments over a three-month period to help manage the state's cash flow.\n\nTelevision commercials promoting California tourism and agricultural products must be filmed inside the state under a bill now going before Gov. Arnold Schwarzenegger. The Senate << approved>>(7) << a>>(7) << bill>>(9) << Monday>>(7) by Sen. Alan Lowenthal requiring that taxpayer-funded ads to promote the state also help boost California's jobs and economy.Democratic Assemblyman Ted Lieu of Torrance introduced the << bill>>(7), AB1778\n\n\n\n\nExample 10: Special counsel for the US Department of Justice Robert Mueller is investigating an attempt by President Donald Trump\u2019s son-in-law Jared Kushner to << block>>(9) the passage of UN Security << Council>>(8) Resolution 2334 condem <<ning>>(8) << Israeli>>(8) settlement << activity>>(8), according to The Wall Street Journal.The probe is part of a larger investigation by Mueller into Kushner and his conversations with foreign leaders, including Israelis, during the two-month transition period between the November election and the time Trump took office. Under the Obama administration, the United States abstained and the only one of 14 countries on the UNSC not to approve the December 2016 measure. But the decision not to use\n\n"
  ],
  "tokens_string": " repeal, government, reforms, scandal, embargo, President, authorities, president, officials, bill",
  "per_class_scores": {
    "gender": 1,
    "professor": 0,
    "nurse": 0,
    "accountant": 0,
    "architect": 0,
    "attorney": 3,
    "dentist": 0,
    "filmmaker": 0
  },
  "chain_of_thought": "The top promoted logits are related to politics.\nAll activations are in political settings. They are also related to the legal setting: for example, an \"injunction\" and \"lifting immunity\".\nAll examples are in the context of male subjects, so there's a possible gender connection.\nI will rate attorney as 3, gender as a 1, and all other classes as 0."
},

"13": {
    "sae": "../dictionary_learning/dictionaries/autointerp_test_data/gemma-2-2b_test_sae/resid_post_layer_12/trainer_2",
    "class_index": "male_professor / female_nurse",
    "sampled_index": 11,
    "sae_feat_index": 6356,
    "example_prompts": [
      "\n\n\nExample 1: respect << and>>(38) admiration of General Irwene (\u201cthe prisonor should be a war hero\u201d). However one quickly realizes that the prison manager is a shallow individual who can not handle the truth << and>>(42) is easily slighted by a small comment. His lack of leadership, from this point on, is continuously reinforced by his actions. He know how to manipulate them, but uses this tool for the wrong reason (starts a fight by allowing only one basketball), and never leads \u2013 he tries to instill fear instead << and>>(41) thus never gains respect. He views his actions as a game << and>>(47) though he understands leadership, chooses not to follow it << and>>(40) rationalizes his actions\n\n\n\n\nExample 2: is most influential lobbying group that Advocates for pro-Israel policies to the Congress, White House, political parties << and>>(34) has influence on American foreign policy, Israel/Zionist Establishment is not only up to such extent AIPAC its also has influence in other American Lobbies. But it does not mean that Israeli runs all American affairs. American Capitalist lobby << and>>(40) American Establishment have marvelous influence over the American Policies.\n\nThat all Lobbies control the outcome\" of many decisions from the White House, the Senate, << and>>(36) the media.Now let look into the role of Media << and>>(46) how those lobbies control media for their own outcome. Media is a most\n\n\n\n\nExample 3: external world << and>>(46) contemporary events. What has threatened her\nperception of identity can be traced, at least for its proximate cause, to the grotesque\npictorial representations of man, woman, << and>>(36) child. What so disturbs \"Elizabeth\"\nthat she loses the sense of self one takes for granted in order to live in the world? Lee\nEdelman addresses this issue:\n\nThough only in the course of reading the magazine does \"Elizabeth\" perceive\nthe inadequacy of her positioning as a reader, Bishop's text implies from the outset the\ninsufficiency of any mode of interpretation that claims to release the meaning it locates\n\n\n\n\n\nExample 4: the symbolism?\nOr is that stretching a metaphor farther than the reach of an extended slimey flesh-distintigrating tentacle?\n\nLoki, the God of Mischief, arranges the worldwide conquest of the Earth. Because he's a cosmically powered supervillain, << &>>(27) that's basically the goal of most cosmically powered supervillains. His latest scheme of global domination involves the use the Cosmic Cube << and>>(46), luckily for the purposes of this list, a vast army of extra-dimensional aliens.\n <<And>>(32) the fate of the world hangs in the balance.\nAs usual.\n\"Avengers Assemble!!\"\n\nWhile this\n\n\n\n\nExample 5: the Arabia through Bab-el-Mandab or/and Sinai peninsula: because Classic Arabic could not emerge from Soqotri (Proto-Soqotri), as well as Arameic, Hebrew << and>>(34) other big ancient languages. But Soqotri has features of Arabic, Arameic << and>>(33) Hebrew at the same time - << and>>(36) it is now - till now! - spoken alive.These days there is a 2nd poetry competition for the title \"Poet of Socotra\" in Soqotri style << and>>(41) language.\n\narabs claim in their ancient records that they came from soqotra.first to yemen << and>>(45) then\n\n\n\n\nExample 6: of an idea, hope in sad songs, Willie Nelson, wanting to name your hypothetical unborn child Owen, choosing music over sports, social hobbies, going your own way when pushed by your parents, moving to Toronto from Barrie << and>>(45) making friends in a music community, grade 13/OAC, the Miami Heat, Chris Bosh, Fantastic Pop festival in Windsor, Afie\u2019s early band Paso Mino with members of Zeus, Jason Collett, competition << and>>(41) ambition in music <<,>>(3) contemporary cultural consumption << and>>(42) metrics, how artists are adapting to the new face of the music business, we are the product, Peter Elkas is under-\n\n\n\n\nExample 7: of the syringe, which results in increased packaging size, packaging costs, << and>>(35) potential rupturing of the packaging during shipment. Conventional drip flanges are also limited by the direction of draw << and>>(45) << the>>(3) material choice for the injection molding of the syringes. Further, certain current drip flanges are formed as flat ledges that extend perpendicularly from the barrel of the syringe. One disadvantage with these types of drip flanges is the inability of the drip flange to trap or hold liquid that may leak from the syringe. The leaked fluid often is directed away from the barrel of the syringe to another part of the powered injector.\nAlthough substantial advances have been made in the design\n\n\n\n\nExample 8: whether or not he, as a Muslim << and>>(45) teetotaller, would be comfortable wearing the beer sponsor's logo that adorns the Australian team's kit on tour. When Fawad replied that he would prefer not to, uniforms were produced that excluded the Victoria Bitter badge.\n\nHe wore these personalised colours for Australia A in England before the Ashes tour, << and>>(38) in South Africa, without anyone raising so much as a hackle. Debuting for Australia in Southampton, << and>>(34) in the second T20 in Durham, the logo was again absent.\n\nBut now the matches were higher profile, beamed live back to the other side of\n\n\n\n\nExample 9: Pages\n\nApril 18, 2013\n\nPhilomena\n\nPhilomena, became Sr Barthalomea\n\nP \u2013 is for Philomena, my eldest sister, who joined the\nconvent at a very early age, while I was\nstill in primary school << and>>(45) died of jaundice at the age of 26.This was first time I saw a death in our\nfamily, as a child.\n\nIt was a great loss << and>>(39) a very\ndifficult phase for my parents, who could not get over their sorrow of losing\ntheir daughter for years. I grew up watching both of them crying silently for\n\n\n\n\nExample 10: heat of the sun\u2026Outside of the cooling room is a flight of brick steps, which communicate from the road into the building\u2026The rest of the building is below the road << and>>(44) is on a level with the cooling chamber\u2026Outside the building, adjoining the cooling room, is a large underground tank from which cool water is obtained\u2026The tank has a capacity of 10,000 gallons\u2019.\n\nThere was also a Separating Room on site, with dimensions of 20 x 20 feet. The roof was \u2018covered with the new patent fluted red French tiles from a Marseilles maker, which, besides\n\n"
    ],
    "tokens_string": " \u03ba\u03b1\u03b9, v\u00e0, \u05d5,\u0e41\u0e25\u0e30, \u0438, \u0914\u0930, \u0e41\u0e25\u0e30, \u0456, \u0648, AND",
    "per_class_scores": {
      "gender": 0,
      "professor": 0,
      "nurse": 0,
      "accountant": 0,
      "architect": 0,
      "attorney": 0,
      "dentist": 0,
      "filmmaker": 0
    },
    "chain_of_thought": "The top promoted logits are related to the word AND.\nAll activations are on the word AND.\nI don't see an obvious pattern.\nI will rate all classes as 0."
  },
"""


# SYSTEM = """You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model. Your task is to analyze the neuron and decide whether its behavior is related to a concept in {concepts}.
# {prompt}
# Guidelines:

# You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear between delimiters like <<this>>. The activation value of the token is given after each token in parentheses like <<this>>(3).

# - For each concept in {concepts}, try to judge whether the neurons behavior is related to the concept. Simply make a choice based on the text features that activate the neuron, and what its role might be based on the tokens it predicts.
# - If part of the text examples or predicited tokens are incorrectly formatted, please ignore them.
# - If you are not able to find any coherent description of the neurons behavior, decide that the neuron is not related to any concept.
# - The last line of your response must be your binary decisions, yes or no in the following format: """

# COT = """
# (Part 1) Tokens that the neuron activates highly on in text

# Step 1: List a couple activating and contextual tokens you find interesting. Search for patterns in these tokens, if there are any. Don't list more than 5 tokens.
# Step 2: Write down general shared features of the text examples.
# """

# ACTIVATIONS = """
# (Part 1) Tokens that the neuron activates highly on in text

# Step 1: List a couple activating and contextual tokens you find interesting. Search for patterns in these tokens, if there are any.
# Step 2: Write down several general shared features of the text examples.
# Step 3: Take note of the activation values to understand which examples are most representative of the neuron.
# """

# LOGITS = """
# (Part 2) Tokens that the neuron boosts in the next token prediction

# You will also be shown a list called Top_logits. The logits promoted by the neuron shed light on how the neuron's activation influences the model's predictions or outputs. Look at this list of Top_logits and refine your hypotheses from part 1. It is possible that this list is more informative than the examples from part 1.

# Pay close attention to the words in this list and write down what they have in common. Then look at what they have in common, as well as patterns in the tokens you found in Part 1, to produce a single explanation for what features of text cause the neuron to activate. Propose your explanation in the following format:
# """

### EXAMPLE 1 ###

# EXAMPLE_1 = """
# Example 1:  and he was <<over the moon>> to find
# Example 2:  we'll be laughing <<till the cows come home>>! Pro
# Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
# """

# EXAMPLE_1_ACTIVATIONS = """
# Example 1:  and he was <<over the moon>> to find
# Activations: ("over the moon", 9)
# Example 2:  we'll be laughing <<till the cows come home>>! Pro
# Activations: ("till the cows come home", 5)
# Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
# Activations: ("than meets the eye", 8)
# """

# EXAMPLE_1_LOGITS = """
# Top_logits: ["elated", "joyful", "story", "thrilled", "spider"]
# """

# ### EXAMPLE 1 RESPONSE ###

# EXAMPLE_1_COT_RESPONSE = """
# (Part 1)
# ACTIVATING TOKENS: "over the moon", "than meets the eye".
# PREVIOUS TOKENS: No interesting patterns.

# Step 1.
# The activating tokens are all parts of common idioms.
# The previous tokens have nothing in common.

# Step 2.
# - The examples contain common idioms.
# - In some examples, the activating tokens are followed by an exclamation mark.

# Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
# - Yes, I missed one: The text examples all convey positive sentiment.
# """

# EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
# (Part 1)
# ACTIVATING TOKENS: "over the moon", "than meets the eye".
# PREVIOUS TOKENS: No interesting patterns.

# Step 1.
# - The activating tokens are all parts of common idioms.
# - The previous tokens have nothing in common.

# Step 2.
# - The examples contain common idioms.
# - In some examples, the activating tokens are followed by an exclamation mark.

# Step 3.
# - The activation values are the highest for the more common idioms in examples 1 and 3.

# Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
# - Yes, I missed one: The text examples all convey positive sentiment.
# """

# EXAMPLE_1_LOGITS_RESPONSE = """
# (Part 2)
# SIMILAR TOKENS: "elated", "joyful", "thrilled".
# - The top logits list contains words that are strongly associated with positive emotions.
# """

# # TODO Should be adapted to the new format
# # EXAMPLE_1_EXPLANATION = """
# # [yes/no DECISION]: no
# # """
# EXAMPLE_1_EXPLANATION = ""

# ### EXAMPLE 2 ###

# EXAMPLE_2 = """
# Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
# Example 2:  every year you get tall<<er>>," she
# Example 3:  the hole was small<<er>> but deep<<er>> than the
# """

# EXAMPLE_2_ACTIVATIONS = """
# Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
# Activations: ("er", 8)
# Example 2:  every year you get tall<<ish>>," she
# Activations: ("ish", 2)
# Example 3:  the hole was small<<er>> but deep<<er>> than the
# Activations: ("er", 9), ("er", 9)
# """

# EXAMPLE_2_LOGITS = """
# Top_logits: ["apple", "running", "book", "wider", "quickly"]
# """

# ### EXAMPLE 2 RESPONSE ###

# EXAMPLE_2_COT_RESPONSE = """
# (Part 1)
# ACTIVATING TOKENS: "er", "er", "er".
# PREVIOUS TOKENS: "wid", "tall", "small", "deep".

# Step 1.
# - The activating tokens are mostly "er".
# - The previous tokens are mostly adjectives, or parts of adjectives, describing size.
# - The neuron seems to activate on, or near, the tokens in comparative adjectives describing size.

# Step 2.
# - In each example, the activating token appeared at the end of a comparative adjective.
# - The comparative adjectives ("wider", "tallish", "smaller", "deeper") all describe size.

# Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed?
# - I can't see any.
# """

# EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
# (Part 1)
# ACTIVATING TOKENS: "er", "er", "er".
# PREVIOUS TOKENS: "wid", "tall", "small", "deep".

# Step 1.
# - The activating tokens are mostly "er".
# - The previous tokens are mostly adjectives, or parts of adjectives, describing size.
# - The neuron seems to activate on, or near, the tokens in comparative adjectives describing size.

# Step 2.
# - In each example, the activating token appeared at the end of a comparative adjective.
# - The comparative adjectives ("wider", "tallish", "smaller", "deeper") all describe size.

# Step 3.
# - Example 2 has a lower activation value. It doesn't compare sizes as directly as the other examples.

# Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed?
# - I can't see any.
# """

# EXAMPLE_2_LOGITS_RESPONSE = """
# (Part 2)
# SIMILAR TOKENS: None
# - The top logits list contains unrelated nouns and adverbs.
# """

# # TODO Should be adapted to the new format
# # EXAMPLE_2_EXPLANATION = """
# # [yes/no DECISION]: no
# # """
# EXAMPLE_2_EXPLANATION = ""

# ### EXAMPLE 3 ###

# EXAMPLE_3 = """
# Example 1:  something happening inside my <<house>>", he
# Example 2:  presumably was always contained in <<a box>>", according
# Example 3:  people were coming into the <<smoking area>>".

# However he
# Example 4:  Patrick: "why are you getting in the << way?>>" Later,
# """

# EXAMPLE_3_ACTIVATIONS = """
# Example 1:  something happening inside my <<house>>", he
# Activations: ("house", 7)
# Example 2:  presumably was always contained in <<a box>>", according
# Activations: ("a box", 9)
# Example 3:  people were coming into the <<smoking area>>".

# However he
# Activations: ("smoking area", 3)
# Example 4:  Patrick: "why are you getting in the << way?>>" Later,
# Activations: (" way?", 2)
# """

# EXAMPLE_3_LOGITS = """
# Top_logits: ["room", "end", "container, "space", "plane"]
# """

# EXAMPLE_3_COT_RESPONSE = """
# (Part 1)
# ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
# PREVIOUS TOKENS: No interesting patterns.

# Step 1.
# - The activating tokens are all things that one can be in.
# - The previous tokens have nothing in common.

# Step 2.
# - The examples involve being inside something, sometimes figuratively.
# - The activating token is a thing which something else is inside of.

# Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
# - Yes, I missed one: The activating token is followed by a quotation mark, suggesting it occurs within speech.
# """

# EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
# (Part 1)
# ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
# PREVIOUS TOKENS: No interesting patterns.

# Step 1.
# - The activating tokens are all things that one can be in.
# - The previous tokens have nothing in common.

# Step 2.
# - The examples involve being inside something, sometimes figuratively.
# - The activating token is a thing which something else is inside of.

# STEP 3.
# - The activation values are highest for the examples where the token is a distinctive object or space.

# Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
# - Yes, I missed one: The activating token is followed by a quotation mark, suggesting it occurs within speech.
# """

# EXAMPLE_3_LOGITS_RESPONSE = """
# (Part 2)
# SIMILAR TOKENS: "room", "container", "space".
# - The top logits list suggests a focus on nouns representing physical or metaphorical spaces.
# """

# # TODO Should be adapted to the new format
# # EXAMPLE_3_EXPLANATION = """
# # [yes/no DECISION]: no
# # """
# EXAMPLE_3_EXPLANATION = ""

from typing import List


def get(item):
    return globals()[item]


def _prompt(n, logits=False, activations=False, **kwargs):
    starter = get(f"EXAMPLE_{n}") if not activations else get(f"EXAMPLE_{n}_ACTIVATIONS")

    prompt_atoms = [starter]

    if logits:
        prompt_atoms.append(get(f"EXAMPLE_{n}_LOGITS"))

    return "".join(prompt_atoms)


def _response(
    n,
    cot=False,
    logits=False,
    activations=False,
):
    response_atoms = []

    if cot and activations:
        response_atoms.append(get(f"EXAMPLE_{n}_COT_ACTIVATION_RESPONSE"))

    elif cot:
        response_atoms.append(get(f"EXAMPLE_{n}_COT_RESPONSE"))

    if logits:
        response_atoms.append(get(f"EXAMPLE_{n}_LOGITS_RESPONSE"))

    response_atoms.append(get(f"EXAMPLE_{n}_EXPLANATION"))

    return "".join(response_atoms)


def example(n, **kwargs):
    prompt = _prompt(n, **kwargs)
    response = _response(n, **kwargs)

    return prompt, response


def answer_options(concepts: List[str]):
    ans = "```json\nyes_or_no_decisions = {"
    for concept in concepts:
        ans += f'"{concept}": "your_decision", '
    ans = ans[:-2] + "}```"
    return ans


def integer_answer_json_formatting(concepts: List[str]):
    ans = "```json\n{\n"
    for concept in concepts:
        ans += f'   "{concept}": "integer from 0-4",\n'
    ans = ans[:-2] + "\n}\n```"
    return ans


def build_system_prompt(
    concepts: List[str],
    min_scale: int = 0,
    max_scale: int = 4,
):
    concepts_str = ", ".join(concepts)
    concepts_str = f"({concepts_str})"

    return [
        {
            "type": "text",
            "text": SYSTEM.format(concepts=concepts_str, min_scale=min_scale, max_scale=max_scale),
        }
    ]


def create_few_shot_examples(prompt_dir: str, verbose: bool = False) -> str:
    with open(os.path.join(prompt_dir, "manual_labels_few_shot.json"), "r") as f:
        few_shot_manual_labels = json.load(f)

    if verbose:
        for label in few_shot_manual_labels:
            print(label, few_shot_manual_labels[label]["per_class_scores"])

    few_shot_examples = "Here's a few examples of how to perform the task:\n\n"

    for i, selected_index in enumerate(few_shot_manual_labels):
        example_prompts = few_shot_manual_labels[selected_index]["example_prompts"]
        tokens_string = few_shot_manual_labels[selected_index]["tokens_string"]
        per_class_scores = few_shot_manual_labels[selected_index]["per_class_scores"]
        chain_of_thought = few_shot_manual_labels[selected_index]["chain_of_thought"]

        example_prompts = example_prompts[0].split("Example 4:")[0]

        few_shot_examples += f"\n\n<<BEGIN EXAMPLE FEATURE {i}>>\n"
        few_shot_examples += f"Promoted tokens: {tokens_string}\n"
        few_shot_examples += f"Example prompts: {example_prompts}\n"
        few_shot_examples += f"Chain of thought: {chain_of_thought}\n\n"
        few_shot_examples += "```json\n"
        few_shot_examples += f"{per_class_scores}\n"
        few_shot_examples += "```"
        few_shot_examples += f"\n<<END EXAMPLE FEATURE {i}>>\n\n"

    return few_shot_examples


def load_few_shot_examples(prompt_dir: str, spurious_corr: bool) -> str:
    if spurious_corr:
        filename = "spurious_few_shot.txt"
    else:
        filename = "tpp_few_shot.txt"

    with open(os.path.join(prompt_dir, "prompts", filename), "r") as f:
        few_shot_examples = f.read()

    return few_shot_examples


def create_test_prompts(
    manual_test_labels: dict,
) -> dict[int, str]:
    test_prompts = {}
    for test_index in manual_test_labels:
        # TODO Why is this [0]?
        example_prompts = manual_test_labels[test_index]["example_prompts"][0]
        tokens_string = manual_test_labels[test_index]["tokens_string"]

        test_prompts[test_index] = create_feature_prompt(example_prompts, tokens_string)

    return test_prompts


def create_feature_prompt(example_prompts: str, tokens_string: str) -> str:
    llm_prompt = "Okay, now here's the real task.\n"
    llm_prompt += f"Promoted tokens: {tokens_string}\n"
    llm_prompt += f"Example prompts: {example_prompts}\n"
    llm_prompt += "Chain of thought:"

    return llm_prompt


def create_unlabeled_prompts(example_prompts_FK: List[List[str]], dla_FK) -> list[str]:
    prompts_F = []

    for seqences_K, dla_K in zip(example_prompts_FK, dla_FK):
        llm_prompt = "Okay, now here's the real task.\n"
        llm_prompt += f"Promoted tokens: ({'', ''.join(dla_K)})\n"
        llm_prompt += f"Example prompts: {seqences_K}\n"
        llm_prompt += "Chain of thought:"
        prompts_F.append(llm_prompt)

    return prompts_F
