package com.dsa.datastructures.trie;


import java.util.HashMap;
import java.util.Map;

public class Trie {

    static class TrieNode {

        String w;
        Map<String, TrieNode> nodes;
        boolean isEnd;

        public TrieNode() {
            this.isEnd = false;
            this.nodes = new HashMap<>();
        }

        public TrieNode(String w) {
            this.isEnd = false;
            this.nodes = new HashMap<>();
            this.w = w;
        }
    }

    TrieNode root = new TrieNode();

    public void add(String[] word) {
        TrieNode cur = root;
        for (String w : word) {
            if (!cur.nodes.containsKey(w)) {
                cur.nodes.put(w, new TrieNode(w));
            }
            cur = cur.nodes.get(w);
        }
        cur.isEnd = true;
    }

    public boolean contains(String[] word) {
        TrieNode cur = getNode(word);
        if (cur != null) {
            return cur.isEnd;
        }
        return false;
    }

    public boolean remove(String[] word) {
        if (contains(word)) {
            removeDFS(root, 0, word);
            return true;
        }
        return false;
    }

    private boolean removeDFS(TrieNode cur, int idx, String[] word) {
        if (idx == word.length) {
            if (cur.isEnd) {
                cur.isEnd = false;
                return cur.nodes.isEmpty();
            }
            return false;
        }
        if (!cur.nodes.containsKey(word[idx])) {
            return false;
        }
        boolean res = removeDFS(cur.nodes.get(word[idx]), idx + 1, word);
        if (res) {
            cur.nodes.remove(word[idx]);
            return cur.nodes.isEmpty() && !cur.isEnd; // tricky: need to check if it is end of another word
        }
        return false;
    }

    private TrieNode getNode(String[] word) {
        TrieNode cur = root;
        for (String w : word) {
            if (!cur.nodes.containsKey(w)) {
                return null;
            }
            cur = cur.nodes.get(w);
        }
        return cur;
    }


    
}
