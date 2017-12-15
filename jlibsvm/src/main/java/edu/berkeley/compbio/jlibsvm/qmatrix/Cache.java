package edu.berkeley.compbio.jlibsvm.qmatrix;

import edu.berkeley.compbio.jlibsvm.SvmException;

/**
 * Kernel Cache
 * <p/>
 * l is the number of total data items size is the cache size limit in bytes
 *
 * @author <a href="mailto:dev@davidsoergel.com">David Soergel</a>
 * @version $Id$
 */

class Cache {
// ------------------------------ FIELDS ------------------------------

  private final int l;
  private long size;

  private final head_t[] head;
  private head_t lru_head;

// --------------------------- CONSTRUCTORS ---------------------------

  Cache(int l_, long size_) {
    if (size_ < 0) {
      throw new SvmException("Cache size < 0");
    }
    l = l_;
    size = size_;
    head = new head_t[l];
    for (int i = 0; i < l; i++) {
      head[i] = new head_t();
    }
    size /= 4;
    size -= l * (16L / 4L);// sizeof(head_t) == 16
    size = Math.max(size, 2 * (long) l);// cache must be large enough for two columns
    lru_head = new head_t();
    lru_head.next = lru_head.prev = lru_head;
  }

// -------------------------- OTHER METHODS --------------------------

  // request data [0,len)
  // return some position p where [p,len) need to be filled
  // (p >= len if nothing needs to be filled)
  // java: simulate pointer using single-element array

  int get_data(int index, double[][] data, int len) {
    head_t h = head[index];
    if (h.len > 0) {
      lru_delete(h);
    }
    int more = len - h.len;

    if (more > 0) {
      // free old space
      while (size < more) {
        head_t old = lru_head.next;
        lru_delete(old);
        size += old.len;
        old.data = null;
        old.len = 0;
      }

      // allocate new space
      double[] new_data = new double[len];
      if (h.data != null) {
        System.arraycopy(h.data, 0, new_data, 0, h.len);
      }
      h.data = new_data;
      size -= more;
      do {
        int temp = h.len;
        h.len = len;
        len = temp;
      }
      while (false);
    }

    lru_insert(h);
    data[0] = h.data;
    return len;
  }

  private void lru_delete(head_t h) {
    // delete from current location
    h.prev.next = h.next;
    h.next.prev = h.prev;
  }

  private void lru_insert(head_t h) {
    // insert to last position
    h.next = lru_head;
    h.prev = lru_head.prev;
    h.prev.next = h;
    h.next.prev = h;
  }

  void swap_index(int i, int j) {
    if (i == j) {
      return;
    }

    if (head[i].len > 0) {
      lru_delete(head[i]);
    }
    if (head[j].len > 0) {
      lru_delete(head[j]);
    }
    do {
      double[] temp = head[i].data;
      head[i].data = head[j].data;
      head[j].data = temp;
    }
    while (false);
    do {
      int temp = head[i].len;
      head[i].len = head[j].len;
      head[j].len = temp;
    }
    while (false);
    if (head[i].len > 0) {
      lru_insert(head[i]);
    }
    if (head[j].len > 0) {
      lru_insert(head[j]);
    }

    if (i > j) {
      do {
        int temp = i;
        i = j;
        j = temp;
      }
      while (false);
    }
    for (head_t h = lru_head.next; h != lru_head; h = h.next) {
      if (h.len > i) {
        if (h.len > j) {
          do {
            double temp = h.data[i];
            h.data[i] = h.data[j];
            h.data[j] = temp;
          }
          while (false);
        } else {
          // give up
          lru_delete(h);
          size += h.len;
          h.data = null;
          h.len = 0;
        }
      }
    }
  }

// -------------------------- INNER CLASSES --------------------------

  private static final class head_t {
// ------------------------------ FIELDS ------------------------------

    head_t prev, next;// a circular list
    double[] data;
    int len;// data[0,len) is cached in this entry
  }
}
