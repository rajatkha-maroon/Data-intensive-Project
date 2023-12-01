#include <bits/stdc++.h>

using namespace std;

int main(void)
{
	int cnt = 0;

	for (int i = 0; i < 8; i+=4) {
		for (int j = 0; j < 12; j+=4) {
			for (int k = 0; k < 16; k+=4) {
				for (int l = 0; l < 16; l+=4) {
					cout << "Midplane " << dec << cnt++ << ": ";
					cout << hex << uppercase << i << hex << uppercase << j << hex << uppercase << k << hex << uppercase << l << hex << uppercase << 0 << "-";
					cout << hex << uppercase << i + 3 << hex << uppercase << j + 3 << hex << uppercase << k + 3 << hex << uppercase << l + 3 << hex << uppercase << 1 << endl;
				}
			}
		}
	}
}
