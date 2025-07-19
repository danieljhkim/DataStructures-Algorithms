package com.dsa.leetcode.trie;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TrieSolutions {


    // 1233. Remove Sub-Folders from the Filesystem
    class P1233 {

        List<String> ans = new ArrayList<>();

        static class TrieNode {

            String w;
            Map<String,TrieNode> node;
            boolean isEnd;

            public TrieNode() {
                this.isEnd = false;
                this.node = new HashMap<>();
            }

            public TrieNode(String w) {
                this.isEnd = false;
                this.node = new HashMap<>();
            }
        }

        void dfs(TrieNode trieNode, List<String> files) {
            if (trieNode.isEnd) {
                ans.add(String.join("/", files));
                return;
            }
            for (String key : trieNode.node.keySet()) {
                files.add(key);
                dfs(trieNode.node.get(key), files);
                files.removeLast();
            }
        }

        public List<String> removeSubfolders(String[] folder) {

            TrieNode root = new TrieNode();

            for (String f : folder) {
                TrieNode cur = root;
                String[] files = f.split("/");
                for (int i = 0; i < files.length; i++) {
                    String file = files[i];
                    if (cur.node.containsKey(file)) {
                        cur = cur.node.get(file);
                    } else {
                        cur.node.put(file, new TrieNode(file));
                        cur = cur.node.get(file);
                    }
                }
                cur.isEnd = true;
            }

            List<String> files = new ArrayList<>();
            dfs(root, files);
            return ans;
        }
    }
}
