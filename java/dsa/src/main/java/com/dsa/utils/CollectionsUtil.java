package com.dsa.utils;

import java.util.HashSet;
import java.util.Set;

public class CollectionsUtil {

    public static Set<Character> arrayToSet(char...array) {
        Set<Character> wset = new HashSet<>();
        for (char c : array) {
            wset.add(c);
        }
        return wset;
    }

    public static Set<Integer> arrayToSet(int...array) {
        Set<Integer> wset = new HashSet<>();
        for (int c : array) {
            wset.add(c);
        }
        return wset;
    }

    public static Set<String> arrayToSet(String...array) {
        Set<String> wset = new HashSet<>();
        for (String c : array) {
            wset.add(c);
        }
        return wset;
    }
}
