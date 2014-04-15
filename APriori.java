import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

/**
 * 
 * A-Priori algorithm (frequent item set mining and association rule learning)
 * works with pairs and triples
 * 
 * @author Douglas Fernando da Silva - doug.fernando@gmail.com
 * 
 */
public class APriori {

	private String filePath;
	private static final int MIN_SUPPORT = 100;
	private static final int ITEMS_TO_PRINT = 20;
	private static final boolean PRINT_PROGRESS = false;

	public static void main(String[] args) throws IOException {
		try {
			APriori ap = new APriori(args[0]);
			ap.executePairs();
			ap.executeTriples();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * @param filePath
	 *            input file path
	 */
	public APriori(String filePath) {
		this.filePath = filePath;
	}

	/**
	 * A-Priori algorithm for triples
	 */
	private void executeTriples() throws IOException {
		// Frequent individual items
		CountSingleItemsFrequencyTask task1 = new CountSingleItemsFrequencyTask();
		executeTask(task1);
		Map<String, Integer> freqIndividualItems = task1.getIndividualItems();
		if (PRINT_PROGRESS) System.out.println(new Date() + " - Found frequent items. Total:" + freqIndividualItems.size());

		// Frequent pairs
		CountPairsFrequencyTask task2 = new CountPairsFrequencyTask(freqIndividualItems);
		executeTask(task2);
		Set<Pair> frequentPairs = task2.getFrequentPairs();
		if (PRINT_PROGRESS) System.out.println(new Date() + " - Found frequent pairs. Total:" + frequentPairs.size());

		// Frequent triples
		CountTriplesFrequencyTask task3 = new CountTriplesFrequencyTask(frequentPairs);
		executeTask(task3);
		Set<Triple> frequentTriples = task3.getFrequentTriple();

		if (PRINT_PROGRESS) System.out.println(new Date() + " - Found frequent triples. Total:" + frequentTriples.size());

		// Generate rules
		Set<TripleRule> tripleRules = generateTripleRules(frequentTriples);
		calculateTripleConfidence(frequentPairs, tripleRules);

		printTop5Rules(tripleRules.toArray(new Rule[0]));
	}

	/**
	 * A-Priori algorithm for pairs
	 */
	private void executePairs() throws IOException {
		// Frequent individual items
		CountSingleItemsFrequencyTask task1 = new CountSingleItemsFrequencyTask();
		executeTask(task1);
		Map<String, Integer> freqIndividualItems = task1.getIndividualItems();
		if (PRINT_PROGRESS) System.out.println(new Date() + " - Found frequent items. Total:" + freqIndividualItems.size());

		// Frequent pairs
		CountPairsFrequencyTask task2 = new CountPairsFrequencyTask(freqIndividualItems);
		executeTask(task2);
		Set<Pair> frequentPairs = task2.getFrequentPairs();
		if (PRINT_PROGRESS) System.out.println(new Date() + " - Found frequent pairs. Total:" + frequentPairs.size());
		
		// Rules from frequent pairs
		Set<PairRule> generatedPairRules = generateRules(frequentPairs);
		calculateConfidence(freqIndividualItems, generatedPairRules);
		if (PRINT_PROGRESS) System.out.println(new Date() + " - Found pair rules. Total:" + generatedPairRules.size());

		printTop5Rules(generatedPairRules.toArray(new Rule[0]));
	}

	/**
	 * Print the top five rules based on their confidence
	 * 
	 * @param rules
	 *            all generated rules
	 */
	private void printTop5Rules(Rule[] rules) {
		Comparator<Rule> comparator = new Comparator<Rule>() {
			@Override
			public int compare(Rule o1, Rule o2) {
				int result = o2.getConfidence().compareTo(o1.getConfidence());
				if (result == 0) {
					result = o1.compareTo(o2);
				}
				
				return result;
			}
		};

		Arrays.sort(rules, comparator);

		int count = 0;
		for (Rule p : rules) {
			if (count < ITEMS_TO_PRINT) {
				System.out.println(p);
				count++;
			} else {
				break;
			}
		}
	}

	/**
	 * Generate rules based on frequent triples
	 */
	private Set<TripleRule> generateTripleRules(Set<Triple> frequentTriples) {
		HashSet<TripleRule> result = new HashSet<APriori.TripleRule>();

		for (Triple t : frequentTriples) {
			result.add(new TripleRule(t.getFirstItem(), t.getSeconditem(), t.getThirdItem(), t.getSupport()));
			result.add(new TripleRule(t.getSeconditem(), t.getThirdItem(), t.getFirstItem(), t.getSupport()));
			result.add(new TripleRule(t.getFirstItem(), t.getThirdItem(), t.getSeconditem(), t.getSupport()));
		}

		return result;
	}

	/**
	 * Generate rules based on frequent pairs
	 */
	private Set<PairRule> generateRules(Set<Pair> frequentPairs) {
		HashSet<PairRule> result = new HashSet<APriori.PairRule>();

		for (Pair p : frequentPairs) {
			result.add(new PairRule(p.getFirstItem(), p.getSeconditem(), p.getSupport()));
			result.add(new PairRule(p.getSeconditem(), p.getFirstItem(), p.getSupport()));
		}

		return result;
	}

	/**
	 * Calculate confidence for triple rules
	 */
	private void calculateTripleConfidence(Set<Pair> frequentPairs, Set<TripleRule> tripleRules) {
		for (TripleRule t : tripleRules) {
			t.calculateConfidence(findPairByValues(frequentPairs, t.getFirstItem(), t.getSeconditem()).getSupport());
		}

	}

	/**
	 * Search a given pair in the set based on the pair values
	 */
	private Pair findPairByValues(Set<Pair> frequentPairs, String firstItem, String secondItem) {
		Pair result = null;

		for (Pair p : frequentPairs) {
			if (p.equals(new Pair(firstItem, secondItem))) {
				result = p;
				break;
			}
		}

		return result;
	}

	/**
	 * Calculate the confidence of pair rules
	 */
	private void calculateConfidence(Map<String, Integer> freqIndividualItems, Set<PairRule> pairRules) {
		PairRule p1 = new PairRule("GRO85051", "FRO40251", 0);
		for (PairRule p : pairRules) {
			if (p.equals(p1)) {
				System.out.println("");
			}
			p.calculateConfidence(freqIndividualItems.get(p.getFirstItem()));
		}
	}

	// DOMAIN CLASSES

	interface Rule {
		Float getConfidence();
		int compareTo(Rule other);
	}

	static class TripleRule extends Triple implements Rule {
		Integer support;

		private Float confidence;

		public Integer getSupport() {
			return support;
		}

		public void calculateConfidence(Integer supportForPair) {
			this.confidence = (float) support / supportForPair;

		}

		public Float getConfidence() {
			return this.confidence;
		}

		public TripleRule(String firstItem, String seconditem, String thirdItem, Integer tripleSupport) {
			super(firstItem, seconditem, thirdItem);
			this.support = tripleSupport;
		}

		@Override
		public String toString() {
			return "(" + this.getFirstItem() + "," + this.getSeconditem() + ") => " + this.getThirdItem()
					+ " | confidence: " + this.confidence;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = super.hashCode();
			result = prime * result + ((this.getFirstItem() == null) ? 0 : this.getFirstItem().hashCode());
			result = prime * result + ((this.getSeconditem() == null) ? 0 : this.getSeconditem().hashCode());
			result = prime * result + ((getThirdItem() == null) ? 0 : getThirdItem().hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (!super.equals(obj))
				return false;
			if (getClass() != obj.getClass())
				return false;
			Triple other = (Triple) obj;
			if (getFirstItem() == null) {
				if (other.getFirstItem() != null)
					return false;
			} else if (!getFirstItem().equals(other.getFirstItem()))
				return false;
			if (getSeconditem() == null) {
				if (other.getSeconditem() != null)
					return false;
			} else if (!getSeconditem().equals(other.getSeconditem()))
				return false;
			if (getThirdItem() == null) {
				if (other.getThirdItem() != null)
					return false;
			} else if (!getThirdItem().equals(other.getThirdItem()))
				return false;
			return true;
		}

		@Override
		public int compareTo(Rule other) {
			TripleRule r = (TripleRule)other;
			int result = getFirstItem().compareTo(r.getFirstItem());
			if (result == 0) result = getSeconditem().compareTo(r.getSeconditem());
			if (result == 0) result = getThirdItem().compareTo(r.getThirdItem());
			
			return result;
		}

	}

	static class PairRule extends Pair implements Rule {
		Integer support;

		private Float confidence;

		public Integer getSupport() {
			return support;
		}

		public void calculateConfidence(Integer supportForFirstItem) {
			this.confidence = (float) support / supportForFirstItem;
		}

		public Float getConfidence() {
			return this.confidence;
		}

		public PairRule(String firstItem, String seconditem, Integer pairSupport) {
			super(firstItem, seconditem, false);
			this.support = pairSupport;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = super.hashCode();
			result = prime * result + ((getFirstItem() == null) ? 0 : getFirstItem().hashCode());
			result = prime * result + ((getSeconditem() == null) ? 0 : getSeconditem().hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (!super.equals(obj))
				return false;
			if (getClass() != obj.getClass())
				return false;
			PairRule other = (PairRule) obj;
			if (getFirstItem() == null) {
				if (other.getFirstItem() != null)
					return false;
			} else if (!getFirstItem().equals(other.getFirstItem()))
				return false;
			if (getSeconditem() == null) {
				if (other.getSeconditem() != null)
					return false;
			} else if (!getSeconditem().equals(other.getSeconditem()))
				return false;
			return true;
		}

		@Override
		public String toString() {
			return this.getFirstItem() + " => " + this.getSeconditem() + " | confidence: " + this.confidence;
		}
		
		@Override
		public int compareTo(Rule other) {
			PairRule r = (PairRule)other;
			int result = getFirstItem().compareTo(r.getFirstItem());
			if (result == 0) result = getSeconditem().compareTo(r.getSeconditem());
			
			return result;
		}
	}

	static class Triple extends Pair {
		private String thirdItem;

		public Triple(String firstItem, String seconditem, String thirdItem) {
			super(firstItem, seconditem);
			this.thirdItem = thirdItem;
		}

		public Triple(String firstItem, String seconditem, String thirdItem, boolean unused) {
			super(firstItem, seconditem, unused);
			this.thirdItem = thirdItem;
		}
		
		public String getThirdItem() {
			return this.thirdItem;
		}

		@Override
		public int hashCode() {
			int result = getFirstItem().hashCode() * getSeconditem().hashCode() * getThirdItem().hashCode()
					^ (getFirstItem().hashCode() + getSeconditem().hashCode() + getThirdItem().hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (getClass() != obj.getClass())
				return false;
			Triple other = (Triple) obj;

			if (this.getFirstItem().equals(other.getFirstItem()) && this.getSeconditem().equals(other.getSeconditem())
					&& this.getThirdItem().equals(other.getThirdItem()))
				return true;

			if (this.getFirstItem().equals(other.getSeconditem()) && this.getSeconditem().equals(other.getFirstItem())
					&& this.getThirdItem().equals(other.getThirdItem()))
				return true;

			if (this.getFirstItem().equals(other.getThirdItem()) && this.getSeconditem().equals(other.getSeconditem())
					&& this.getThirdItem().equals(other.getFirstItem()))
				return true;

			if (this.getFirstItem().equals(other.getFirstItem()) && this.getSeconditem().equals(other.getThirdItem())
					&& this.getThirdItem().equals(other.getSeconditem()))
				return true;

			if (this.getFirstItem().equals(other.getThirdItem()) && this.getSeconditem().equals(other.getFirstItem())
					&& this.getThirdItem().equals(other.getSeconditem()))
				return true;

			if (this.getFirstItem().equals(other.getSeconditem()) && this.getSeconditem().equals(other.getThirdItem())
					&& this.getThirdItem().equals(other.getFirstItem()))
				return true;

			/*
			 * 1 2 3 = 1 2 3 
			 * 1 2 3 = 2 1 3 
			 * 1 2 3 = 3 2 1 
			 * 1 2 3 = 1 3 2 
			 * 1 2 3 = 3 1 2 
			 * 1 2 3 = 2 3 1
			 */

			return false;
		}

		public String toString() {
			return "(" + getFirstItem() + "," + getSeconditem() + "," + getThirdItem() + ") - support: " + getSupport();
		}
	}

	static class Pair {
		private String firstItem;
		private String seconditem;
		private Integer support = 0;

		public void increaseSupport() {
			this.support++;
		}

		public Integer getSupport() {
			return this.support;
		}

		public String getFirstItem() {
			return firstItem;
		}

		public String getSeconditem() {
			return seconditem;
		}

		public Pair(String firstItem, String seconditem) {
			if (firstItem.compareTo(seconditem) > 0) {
				this.seconditem = firstItem;
				this.firstItem = seconditem;
			} else {
				this.firstItem = firstItem;
				this.seconditem = seconditem;
			}
		}

		public Pair(String firstItem, String seconditem, boolean unused) {
			this.firstItem = firstItem;
			this.seconditem = seconditem;
		}
		
		@Override
		public int hashCode() {
			int result = firstItem.hashCode() * seconditem.hashCode() ^ (firstItem.hashCode() + seconditem.hashCode());

			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Pair other = (Pair) obj;

			if (this.firstItem.equals(other.firstItem) && this.seconditem.equals(other.seconditem))
				return true;
			if (this.firstItem.equals(other.seconditem) && this.seconditem.equals(other.firstItem))
				return true;

			return false;
		}

		public String toString() {
			return "(" + getFirstItem() + "," + getSeconditem() + ") - support: " + getSupport();
		}
	}

	// COMMAND CLASSES

	/**
	 * Template for processing line by line of the input file
	 */
	public void executeTask(Task task) throws IOException {
		FileReader fileReader = new FileReader(filePath);
		BufferedReader br = new BufferedReader(fileReader);
		String line;
		while ((line = br.readLine()) != null) {
			task.execute(line);
		}
		br.close();
	}

	interface Task {
		void execute(String line);
	}

	/**
	 * Task to iterate over the input file and count the support of the frequent
	 * pairs
	 */
	static class CountPairsFrequencyTask implements Task {
		private HashMap<Pair, Pair> items;
		private Map<String, Integer> frequentSingleItems;
		private Integer count = 0;

		public CountPairsFrequencyTask(Map<String, Integer> frequentSingleItems) {
			this.items = new HashMap<Pair, Pair>();
			this.frequentSingleItems = frequentSingleItems;
		}

		public Set<Pair> getFrequentPairs() {
			Set<Pair> result = new HashSet<APriori.Pair>();

			for (Entry<Pair, Pair> p : items.entrySet()) {
				if (p.getKey().getSupport() > MIN_SUPPORT) {
					result.add(p.getKey());
				}
			}

			return result;
		}

		private ArrayList<Pair> generatePotentialPairs(String[] items) {
			ArrayList<Pair> result = new ArrayList<Pair>();

			for (int i = 0; i < items.length; i++) {
				for (int j = i + 1; j < items.length; j++) {
					if (frequentSingleItems.containsKey(items[i]) && frequentSingleItems.containsKey(items[j]))
						result.add(new Pair(items[i], items[j]));
				}
			}

			return result;
		}

		@SuppressWarnings("unused")
		@Override
		public void execute(String line) {
			ArrayList<Pair> potentialPairs = generatePotentialPairs(line.split("\\s+"));

			for (Pair p : potentialPairs) {
				if (items.containsKey(p)) {
					items.get(p).increaseSupport();
				} else {
					p.increaseSupport();
					items.put(p, p);
				}
			}

			count++;

			if (count % 1000 == 0 && PRINT_PROGRESS) {
				System.out.println(new Date() + " - Line:" + count + " processed! - PAIRS");
			}
		}

	}

	/**
	 * Task to iterate over the input file and count the support of the frequent
	 * triple
	 */
	static class CountTriplesFrequencyTask implements Task {
		private HashMap<Triple, Triple> items;
		private Map<Pair, Pair> frequentPairItems;
		private Integer count = 0;

		// private static final Integer NUM_OF_LINES = 31101;

		public CountTriplesFrequencyTask(Set<Pair> frequentPairItems) {
			this.items = new HashMap<Triple, Triple>();
			this.frequentPairItems = new HashMap<Pair, Pair>();
			for (Pair p : frequentPairItems) {
				this.frequentPairItems.put(p, p);
			}
		}

		public Set<Triple> getFrequentTriple() {
			Set<Triple> result = new HashSet<Triple>();

			for (Entry<Triple, Triple> p : items.entrySet()) {
				if (p.getKey().getSupport() > MIN_SUPPORT) {
					result.add(p.getKey());
				}
			}

			return result;
		}

		private ArrayList<Triple> generatePotentialTriples(String[] items) {
			ArrayList<Triple> result = new ArrayList<Triple>();
			
			for (int i = 0; i < items.length; i++) {
				for (int j = i + 1; j < items.length; j++) {
					for (int k = j + 1; k < items.length; k++) {
						Pair p1 = new Pair(items[i], items[j]);
						Pair p2 = new Pair(items[j], items[k]);
						Pair p3 = new Pair(items[i], items[k]);
												
						if (frequentPairItems.containsKey(p1) || frequentPairItems.containsKey(p2)
								|| frequentPairItems.containsKey(p3)) {
							result.add(new Triple(items[i], items[j], items[k]));
						}
					}
				}
			}
			
			return result;
		}

		@SuppressWarnings("unused")
		@Override
		public void execute(String line) {
			ArrayList<Triple> potentialTriple = generatePotentialTriples(line.split("\\s+"));
			
			for (Triple t : potentialTriple) {
				if (items.containsKey(t)) {
					items.get(t).increaseSupport();
				} else {
					t.increaseSupport();
					items.put(t, t);
				}
			}

			count++;

			if (count % 1000 == 0 && PRINT_PROGRESS) {
				System.out.println(new Date() + " - Line:" + count + " processed! - TRIPLES");
			}
		}
	}

	/**
	 * Task to iterate over the input file and count the support of the frequent
	 * items
	 */
	static class CountSingleItemsFrequencyTask implements Task {
		HashMap<String, Integer> items = new HashMap<String, Integer>();

		@Override
		public void execute(String line) {
			for (String token : line.split("\\s+")) {
				Integer finalValue = 1;

				if (items.containsKey(token)) {
					Integer currentValue = items.get(token);
					finalValue = currentValue + 1;
				}

				items.put(token, finalValue);
			}
		}

		public Map<String, Integer> getIndividualItems() {
			HashMap<String, Integer> result = new HashMap<String, Integer>();

			for (Map.Entry<String, Integer> entry : this.items.entrySet()) {
				if (entry.getValue() > MIN_SUPPORT) {
					result.put(entry.getKey(), entry.getValue());
				}
			}

			return result;
		}

	}
}
