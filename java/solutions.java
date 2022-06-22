import java.util.List;

class Solution {

	public int numberOfBeams(String[] bank) {
		int sum = 0;
		List<Integer> ar = new ArrayList<Integer>();
		if(bank.length == 1) return 0;
		for(String row : bank) {
			int beams = 0;
			for(char c : row.toCharArray()) {
				if(c == '1') {
					beams++;
				}
			}
			if(beams > 0) {
				ar.add(beams);
			}
		}
		int prev = 0;
		for(int j=0; j<ar.size()-1; j++) {
			sum += ar.get(j) * ar.get(j+1);
		}
		return sum;
	}
}