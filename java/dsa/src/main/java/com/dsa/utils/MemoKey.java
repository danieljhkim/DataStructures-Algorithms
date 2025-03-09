package com.dsa.utils;

import java.util.Objects;

public class MemoKey {
    int r, c;

    public MemoKey(int r, int c) {
        this.r = r;
        this.c = c;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof MemoKey)) return false;
        MemoKey key = (MemoKey) o;
        return r == key.r && c == key.c;
    }

    @Override
    public int hashCode() {
        return Objects.hash(r, c);
    }
}
