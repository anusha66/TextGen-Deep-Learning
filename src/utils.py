import json

def dump_summaries(filename):
	with open(filename, 'r') as f:
		data = json.load(f)
		for d in data:
			summary_list = d['summary']
			summary_text = " ".join(summary_list)
			print(str(summary_text))


if __name__ == "__main__":
	dump_summaries('../data/rotowire/test.json')