#include<bits/stdc++.h>
using namespace std;
#define lli long long int
#define llu unsigned long long int
#define ld long double
#define nl "\n"
#define fastinput ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
//----------------------------------------------------------------///
class Solution {
public:
	void firstNegativeNUmber(vector<int>&nums,int k) {
		int i=0,j=0;
		int n=nums.size();
		vector<int>ans;
		list<int>ls;
		while(j<n) {
			if(nums[j]<0) {
				ls.push_back(nums[j]);
			}
			if(j-i+1<k) {
				j++;
			}
			else if((j-i+1)==k) {
				if(ls.size()==0) {
					ans.push_back(0);
				}
				else {
					ans.push_back(ls.front());
				}
				if(nums[i]<0) {
					ls.pop_front();
				}
				i++,j++;
			}
		}
		for(int i=0;i<ans.size();i++) {
			cout<<ans[i]<<" ";
		}
	}
};
int main(){
#ifndef ONLINE_JUDGE
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
#endif
	Solution *p=new Solution();
	vector<int>arr{-8,2,3,6,-10};
	int k=3;
	p->firstNegativeNUmber(arr,k);	
    return 0;

}

