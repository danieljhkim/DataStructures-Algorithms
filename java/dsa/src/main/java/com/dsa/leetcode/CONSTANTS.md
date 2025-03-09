int[][] dirs={{0,1},{0,-1},{-1,0},{1,0}};

if (r >= 0 && c >= 0 && r < R && c < C)

int[][] distance = new int[grid.length][grid[0].length];
boolean[][] visited = new boolean[grid.length][grid[0].length];

for (int[] row: distance) {
    Arrays.fill(row, Integer.MAX_VALUE);
}

Queue <int[]> queue = new PriorityQueue <> ((a, b) -> a[2] - b[2]);
