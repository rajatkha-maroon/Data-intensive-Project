#include <bits/stdc++.h>

int main(void) {
	for (int i = 1; i <= 12; i++) {
		for (int j = 1; j <= 31; j++) {
			char s[1024];
			snprintf(s, sizeof(s), "wget https://ftp.mcs.anl.gov/pub/darshan/data/intrepid-iostat/summaries/weekly-2011-%.2d-%.2d.gz", i, j);
			system(s);
		}
	}
}
